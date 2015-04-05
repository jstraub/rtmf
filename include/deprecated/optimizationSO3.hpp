#pragma once

#include <defines.h>

#include <stdint.h>
#include <string>
#include <vector>
#include <iostream>
#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/QR>
#include <unsupported/Eigen/MatrixFunctions>

// CUDA runtime
#include <cuda_runtime.h>
// Utilities and system includes
#include <nvidia/helper_cuda.h>
#include <timer.hpp>

using namespace Eigen;
using namespace std;

#ifdef WEIGHTED
extern "C" void robustSquaredAngleCostFctGPU(float *h_cost, float *d_cost,
    float *d_x, float* d_weights, uint16_t *d_z, float *d_mu, float sigma_sq, 
    int w, int h);

extern "C" void robustSquaredAngleCostFctAssignmentGPU(float *h_cost, 
    float *d_cost,uint32_t *h_N, uint32_t *d_N, float *d_x, float* d_weights, 
    float * d_errs,
    uint16_t *d_z, float *d_mu, float sigma_sq, int w, int h);

extern "C" void robustSquaredAngleCostFctJacobianGPU(float *h_J, float *d_J,
    float *d_x, float *d_weights, uint16_t *d_z, float *d_mu, float sigma_sq, 
    int w, int h);
#else
extern "C" void robustSquaredAngleCostFctGPU(float *h_cost, float *d_cost,
    float *d_x, uint16_t *d_z, float *d_mu, float sigma_sq, int w, int h);

extern "C" void robustSquaredAngleCostFctAssignmentGPU(float *h_cost, 
    float *d_cost, uint32_t *h_N, uint32_t *d_N, float *d_x, uint16_t *d_z, float *d_mu, 
    float sigma_sq, int w, int h);

extern "C" void robustSquaredAngleCostFctJacobianGPU(float *h_J, float *d_J,
    float *d_x, uint16_t *d_z, float *d_mu, float sigma_sq, int w, int h);
#endif

extern "C" void meanInTpS2GPU(float *h_p, float *d_p, float *h_mu_karch,
    float *d_mu_karch, float *d_q, uint16_t *d_z, float* d_weights,int w, int h);

extern "C" void sufficientStatisticsOnTpS2GPU(float *h_p, float *d_p, 
  float *h_Rnorths, float *d_Rnorths, float *d_q, uint16_t *d_z ,int w, int h,
  float *h_SSs, float *d_SSs);

extern "C" void loadRGBvaluesForMFaxes();

class OptSO3
{
protected:
  const int w_,h_;
  float *d_cost, *d_J, *d_mu_;
  float *h_errs_;
  uint32_t *d_N_;
  float t_max_, dt_;

public:
  float *d_errs_;
  float *d_q_, *d_weights_;
  const float sigma_sq_;
  uint16_t *d_z, *h_z;

  float dtPrep_; // time for preparation (here assignemnts)
  float dtCG_; // time for cunjugate gradient

  uint32_t t_; // timestep
  Matrix3f Rprev_; // previous rotation

  OptSO3(float sigma, float *d_q, int w, int h, float *d_weights =NULL):
    w_(w),h_(h),t_max_(1.0f),dt_(0.1f),
    d_q_(d_q),d_weights_(d_weights),
    sigma_sq_(sigma*sigma), dtPrep_(0.0f), dtCG_(0.0f),
    t_(0), Rprev_(Matrix3f::Identity())
  {init();};

  ~OptSO3();

  double D_KL_axisUnif();

  float *getErrs(int w, int h);

  virtual double conjugateGradientCUDA(Matrix3f& R, uint32_t maxIter=100)
  {
    Timer t0;
    uint32_t N = 0;
    if( fabs(R.determinant()-1.0) > 1e-6)
    {
      cout<<" == renormalizing rotation to get back to det(R) = 1"<<endl;
      R.col(0) = R.col(0)/R.col(0).norm();
      R.col(1) = R.col(1)/R.col(1).norm();
      R.col(2) = R.col(0).cross(R.col(1));
    }
    float res0 = conjugateGradientPreparation_impl(R,N);
    dtPrep_ = t0.toctic("--- association ");
    float resEnd = conjugateGradientCUDA_impl(R,res0, N, maxIter);
    conjugateGradientPostparation_impl(R);
    dtCG_ = t0.toctic("--- conjugateGradientCUDA ");
    t_++; // keep track of timesteps 
    Rprev_ = R; // keep track of previous rotation estimate for rot. velocity computation
    return resEnd;
  };
  
  /* return a skew symmetric matrix from A */
  Matrix3f enforceSkewSymmetry(const Matrix3f &A) const
  {return 0.5*(A-A.transpose());};

  float dtPrep(void) const { return dtPrep_;};
  float dtCG(void) const { return dtCG_;};
protected:
  virtual float conjugateGradientPreparation_impl(Matrix3f& R, uint32_t& N);
  virtual float conjugateGradientCUDA_impl(Matrix3f& R, float res0, uint32_t N, uint32_t maxIter=100);
  virtual void conjugateGradientPostparation_impl(Matrix3f& R){;};
  /* evaluate cost function for a given assignment of npormals to axes */
  virtual float evalCostFunction(Matrix3f& R);
  /* compute Jacobian */
  virtual void computeJacobian(Matrix3f&J, Matrix3f& R, float N);
  /* recompute assignment based on rotation R and return residual as well */
  virtual float computeAssignment(Matrix3f& R, uint32_t& N);
  /* updates G and H from rotation R and jacobian J
   */
  virtual void updateGandH(Matrix3f& G, Matrix3f& G_prev, Matrix3f& H, 
      const Matrix3f& R, const Matrix3f& J, const Matrix3f& M_t_min,
      bool resetH);
  /* performs line search starting at R in direction of H
   * returns min of cost function and updates R, and M_t_min
   */
  virtual float linesearch(Matrix3f& R, Matrix3f& M_t_min, const Matrix3f& H, 
    float N, float t_max=1.0f, float dt=0.1f);
  /* mainly init GPU arrays */
  void init();
  /* copy rotation to device */
  void Rot2Device(Matrix3f& R);
  /* convert a Rotation matrix R to a MF representaiton of the axes */
  void Rot2M(Matrix3f& R, float *mu);
  //deprecated
  void rectifyRotation(Matrix3f& R);
};

