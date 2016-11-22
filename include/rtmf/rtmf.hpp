/* Copyright (c) 2015, Julian Straub <jstraub@csail.mit.edu> Licensed
 * under the MIT license. See the license file LICENSE.
 */
#pragma once

#include <Eigen/Dense>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <mmf/optimizationSO3.hpp>
#include <mmf/optimizationSO3_gd.hpp>
#include <mmf/optimizationSO3_approx.hpp>
#include <mmf/optimizationSO3_approx_gd.hpp>
#include <mmf/optimizationSO3_vmf.hpp>
#include <mmf/optimizationSO3_vmfCF.hpp>
#include <mmf/optimizationSO3_mmfvmf.hpp>

#include <cudaPcl/dirSeg.hpp>

using namespace Eigen;
using namespace std;


struct CfgOptSO3
{
  CfgOptSO3() : tMax(1.0f), dt(0.1f), 
    sigma(5.0f*M_PI/180.f), nCGIter(10),
    pathOut("")
  {};
  float tMax; 
  float dt;
  float sigma;
  uint32_t nCGIter;
  string pathOut;
};

class RealtimeMF : public cudaPcl::DirSeg
{
public:
  RealtimeMF(std::string mode, const CfgOptSO3& cfg,
      const cudaPcl::CfgSmoothNormals& cfgNormals);
  virtual ~RealtimeMF();

  virtual MatrixXf centroids(){ return mfAxes();};
  MatrixXf mfAxes();
  const Matrix3f& cRmf() { return cRmf_;};
  std::vector<Matrix3f> cRmfs() { return cRmfs_;};
  const Eigen::VectorXf& counts() { return optSO3_->counts();};
  double cost() { return residual_;};

  void Reset() { 
    optSO3_->ResetRs(); 
    cRmf_ = Eigen::Matrix3f::Identity(); 
  }

  cv::Mat normalsImg_;
protected:
    const static uint32_t K_MAX = 6;

    double residual_, D_KL_;
    uint32_t nCGIter_;

    ofstream fout_;
    std::string mode_;

    mmf::OptSO3 * optSO3_;
    Matrix3f cRmf_;
    std::vector<Eigen::Matrix3f> cRmfs_;
    double cost_;


    /*
     * runs the actual compute; assumes that normalExtract_ contains
     * the normals on GPU already.
     */
    virtual void compute_();
    /* get lables in input format */
    virtual void getLabels_()
    {
      normalExtract_->uncompressCpu(optSO3_->z().data(),
          optSO3_->z().rows(), z_.data(), z_.rows());
    };

    virtual void scaleDirColors(uint32_t K);
};

// ------------------- impl --------------------------------------
RealtimeMF::RealtimeMF(std::string mode, const CfgOptSO3& cfg,
    const cudaPcl::CfgSmoothNormals& cfgNormals)
  : DirSeg(cfgNormals, cfg.pathOut),
  residual_(0.0), D_KL_(0.0), nCGIter_(cfg.nCGIter),
  fout_((cfg.pathOut+std::string("./stats.log")).data(),ofstream::out),
  mode_(mode), optSO3_(NULL),
  cRmf_(MatrixXf::Identity(3,3))
{
  cout<<"inititalizing optSO3"<<endl
    <<"  mode = "<<mode_<<endl
    <<"  sigma = "<<cfg.sigma<<endl
    <<"  tMax = "<<cfg.tMax << " dt= "<<cfg.dt<<endl
    <<"  nCGIter = "<<cfg.nCGIter<<endl;

  if(mode_.compare("direct") == 0)
  {
    optSO3_ = new mmf::OptSO3(cfg.sigma, 
        cfg.tMax, cfg.dt);
  }else if (mode_.compare("directGD") == 0){
    optSO3_ = new mmf::OptSO3GD();
  }else if (mode_.compare("approx") == 0){
    optSO3_ = new mmf::OptSO3Approx(cfg.sigma,
        cfg.tMax, cfg.dt);
  }else if (mode_.compare("approxGD") == 0){
    optSO3_ = new mmf::OptSO3ApproxGD();
  }else if (mode_.compare("vmf") == 0){
    optSO3_ = new mmf::OptSO3vMF(cfg.sigma,
        cfg.tMax, cfg.dt);
  }else if (mode_.compare("vmfCF") == 0){
    optSO3_ = new mmf::OptSO3vMFCF();
  }else if (mode_.compare("mmfvmf") == 0){
    optSO3_ = new mmf::OptSO3MMFvMF(4);
  }
};

RealtimeMF::~RealtimeMF()
{
  delete optSO3_;
  fout_.close();
};

void RealtimeMF::compute_()
{
  // get compressed normals
  tLog_.toctic(1,2);
  int32_t nComp = 0;
  float* d_nComp = normalExtract_->d_normalsComp(nComp);
  cout<<" -- compressed to "<<nComp<<" normals"<<endl;
//      normalsImg_ = normalExtract_->normalsImg();
  // optimize normals
  tLog_.toctic(2,3); 
  optSO3_->updateExternalGpuNormals(d_nComp,nComp,3,0);
  
  residual_ = optSO3_->conjugateGradientCUDA(cRmf_,nCGIter_);
  cRmfs_ = optSO3_->GetRs();
  if (mode_.compare("mmfvmf") == 0 && cRmfs_.size() == 1)
    cRmf_ = cRmfs_[0];
  std::cout << "have " << cRmfs_.size() << " MFs" << std::endl;
  D_KL_ = optSO3_->D_KL_axisUnif();
  cout<<" -- optimized rotation D_KL to unif "<<D_KL_<<endl
    <<cRmf_<<endl;
  tLog_.toc(3); 
//  tLog_.setDt(3,optSO3_->dtPrep());
//  tLog_.setDt(4,optSO3_->dtCG());
  tLog_.logCycle();
  haveLabels_ = false;
};


MatrixXf RealtimeMF::mfAxes()
{
  MatrixXf mfAx = MatrixXf::Zero(3,6*cRmfs_.size());
  for(uint32_t k=0; k<6*cRmfs_.size(); ++k){
    int j = (k%6)/2; // which of the rotation columns does this belong to
    float sign = (- float((k%6)%2) +0.5f)*2.0f; // sign of the axis
    mfAx.col(k) = sign*cRmfs_[k/6].col(j);
  }
  return mfAx;
};

void RealtimeMF::scaleDirColors(uint32_t Kk)
{
  if (mode_.compare("mmfvmf") == 0 && cRmfs_.size() > 1) {
    Matrix<uint8_t,Dynamic,Dynamic> dirCols(5*6,3);
//    this->dirCols_ << 255,0,0, 255,0,0, 0,255,0, 0,255,0, 0,0,255,
//      0,0,255, 255,20,20, 255,20,20, 20,255,20, 20,255,20, 20,20,255,
//      20,20,255, 255,40,40, 255,40,40, 40,255,40, 40,255,40, 40,40,255,
//      40,40,255;
//  32 ,182,232 tuerkis    
//  232,139,32  orange     
//  255,13 ,255 pink       
//  32,232,59 green       
//  232,65,32 red         
    dirCols << 
      32 ,182,232,
      32 ,182,232,
      32 ,182,232,
      32 ,182,232,
      32 ,182,232,
      32 ,182,232,
      232,139,32 ,
      232,139,32 ,
      232,139,32 ,
      232,139,32 ,
      232,139,32 ,
      232,139,32 ,
      255,13 ,255,
      255,13 ,255,
      255,13 ,255,
      255,13 ,255,
      255,13 ,255,
      255,13 ,255,
      32,232,59,
      32,232,59,
      32,232,59,
      32,232,59,
      32,232,59,
      32,232,59,
      232,65,32,   
      232,65,32,   
      232,65,32,   
      232,65,32,   
      232,65,32,   
      232,65,32;
    this->dirCols_ = dirCols.topRows(6*Kk);
    this->K_=6*Kk;
//    std::cout << this->dirCols_.cast<float>() << std::endl;
  } else {
    this->dirCols_ = Matrix<uint8_t,Dynamic,Dynamic>(6,3);
    this->dirCols_ << 255,0,0,
      255,0,0,
      0,255,0,
      0,255,0,
      0,0,255,
      0,0,255;
    this->K_=6;
  }
}
