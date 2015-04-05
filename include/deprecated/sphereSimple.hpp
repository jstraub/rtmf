#pragma once 

#include <stdint.h>
#include <iostream>
#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/QR>
#include <unsupported/Eigen/MatrixFunctions>

using namespace Eigen;

//TODO: use the other sphere class
class SphereSimple
{
  public:
  SphereSimple()
  {}
  ~SphereSimple()
  {}

  /* normal in tangent space around p rotate to the north pole
   * -> the third dimension will always be 0
   * -> return only first 2 dims
   */
  MatrixXf Log_p_2D(const Matrix<float,3,1>& p, const MatrixXf& q)
  {
    return rotate_p2north(p,Log_p(p,q));
  }

  /* rotate points x in tangent plane around north pole down to p
   */
  MatrixXf rotate_north2p(const Matrix<float,3,1>& p, const MatrixXf& xNorth)
  {
    Matrix3f northR = this->north_R_TpS2(p);
    //cout<<"northR"<<endl<<northR<<endl;
    if(xNorth.cols() == 2)
    {
      MatrixXf x(xNorth.rows(),3);
      x = xNorth * northR.transpose().topRows<2>();
      return x;
    }else if (xNorth.rows() == 2){
      MatrixXf x(3,xNorth.cols());
      x = northR.leftCols<2>() * xNorth;
      return x;
    }else{
      assert(false);
    }
  }

  /* rotate points x in tangent plane around p to north pole and 
   * return 2D coordinates
   */
  MatrixXf rotate_p2north(const Matrix<float,3,1>& p, const MatrixXf& x)
  {
    Matrix3f northR = this->north_R_TpS2(p);
    if(x.cols() == 3)
    {
      //MatrixXf xNorth(x.rows(),2);
      assert((x.row(0)*northR.transpose())(2) < 1e-6);
      return (x * northR.transpose()).leftCols<2>();
    }else if (x.rows() == 3){
//#ifndef NDEBUG
//      cout<< (northR * x.col(0)).transpose()<<endl;
//#endif 
      assert((northR * x.col(0))(2) < 1e-3);
      return (northR * x).topRows<2>();
    }else{
      assert(false);
    }
  }

  /* compute rotation from TpS^2 to north pole on sphere
   */
  Matrix3f north_R_TpS2(const Matrix<float,3,1>& p)
  {
    Matrix<float,3,1> north;
    north << 0.f,0.f,1.f;
    Eigen::Quaternion<float> northQ_TpS2;
    northQ_TpS2.setFromTwoVectors(p,north);
    return northQ_TpS2.toRotationMatrix().cast<float>();
  }

  MatrixXf Log_p(const Matrix<float,3,1>& p, const MatrixXf& q)
  {
    MatrixXf x(q.rows(),q.cols());
    if(q.cols() == 3)
    {
      for (uint32_t i=0; i<q.rows(); ++i)
      {
        float dot = max(-1.0f,min(1.0f,q.row(i).dot(p)));
        float theta = acos(dot);
        float sinc;
        if(theta < 1.e-8)
          sinc = 1.0f;
        else
          sinc = theta/sin(theta);
        x.row(i) = (q.row(i)-p.transpose()*dot)*sinc;
      }
    }else if (q.rows() == 3)
    {
      for (uint32_t i=0; i<q.cols(); ++i)
      {
        float dot = max(-1.0f,min(1.0f,p.dot(q.col(i))));
        float theta = acos(dot);
        float sinc;
        if(theta < 1.e-8)
          sinc = 1.0f;
        else
          sinc = theta/sin(theta);
        x.col(i) = (q.col(i)-p*dot)*sinc;
      }
    }else{
      assert(false);
    }
    return x;
  }

  MatrixXf Exp_p(const Matrix<float,3,1>& p, const MatrixXf& x)
  {
//    assert(p.cols ==1);
    MatrixXf q(x.rows(),x.cols());
     
    for (uint32_t i=0; i<x.cols(); ++i){
      float theta_i = x.col(i).norm();
      //cout<<"theta "<<theta_i<<endl;
      if (theta_i < 1e-10)
        q.col(i) = p + x.col(i);
      else
        q.col(i) = p*cos(theta_i) + x.col(i)/theta_i *sin(theta_i);
      //cout<<q.col(i).transpose()<<endl;
    }
    return q;
  }

  protected:
};


