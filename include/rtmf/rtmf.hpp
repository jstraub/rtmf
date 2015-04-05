/* Copyright (c) 2015, Julian Straub <jstraub@csail.mit.edu> Licensed
 * under the MIT license. See the license file LICENSE.
 */
#pragma once

#include <Eigen/Dense>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <jsCore/timerLog.hpp>

#include <mmf/optimizationSO3.hpp>
#include <mmf/optimizationSO3_approx.hpp>
#include <mmf/optimizationSO3_vmf.hpp>

#include <cudaPcl/normalExtractSimpleGpu.hpp>
#include <cudaPcl/depthGuidedFilter.hpp>

using namespace Eigen;
using namespace std;


struct CfgOptSO3
{
  CfgOptSO3() : tMax(1.0f), dt(0.1f), 
    sigma(5.0f*M_PI/180.f), nCGIter(10){};
  float tMax; 
  float dt;
  float sigma;
  uint32_t nCGIter;
};

void projectDirections(cv::Mat& I, const MatrixXf&
    dirs, double f_d, const Matrix<uint8_t,Dynamic,Dynamic>& colors)
{
  double scale = 0.1;
  VectorXf p0(3); p0 << 0.35,0.25,1;
  double u0 = p0(0)/p0(2)*f_d + 320.;
  double v0 = p0(1)/p0(2)*f_d + 240.;
  for(uint32_t k=0; k < dirs.cols(); ++k)
  {
    VectorXf p1 = p0 + dirs.col(k)*scale;
    double u1 = p1(0)/p1(2)*f_d + 320.;
    double v1 = p1(1)/p1(2)*f_d + 240.;
    cv::line(I, cv::Point(u0,v0), cv::Point(u1,v1),
        CV_RGB(colors(k,0),colors(k,1),colors(k,2)), 2, CV_AA);

    double arrowLen = 10.;
    double angle = atan2(v1-v0,u1-u0);

    double ru1 = u1 - arrowLen*cos(angle + M_PI*0.25);
    double rv1 = v1 - arrowLen*sin(angle + M_PI*0.25);
    cv::line(I, cv::Point(u1,v1), cv::Point(ru1,rv1),
        CV_RGB(colors(k,0),colors(k,1),colors(k,2)), 2, CV_AA);
    ru1 = u1 - arrowLen*cos(angle - M_PI*0.25);
    rv1 = v1 - arrowLen*sin(angle - M_PI*0.25);
    cv::line(I, cv::Point(u1,v1), cv::Point(ru1,rv1),
        CV_RGB(colors(k,0),colors(k,1),colors(k,2)), 2, CV_AA);
  }
  cv::circle(I, cv::Point(u0,v0), 2, CV_RGB(0,0,0), 2, CV_AA);
};


class RealtimeMF 
{
public:
  RealtimeMF(std::string mode, const CfgOptSO3& cfgOptSO3,
      const cudaPcl::CfgSmoothNormals& cfgNormals);
  virtual ~RealtimeMF();

  /*
   * compute MF from depth image stored on the CPU
   */
  void compute(const cv::Mat& depth) 
    {compute((uint16_t*)depth.data, depth.cols, depth.rows);};
  void compute(const uint16_t* depth, uint32_t w, uint32_t h);

  /*
   * compute MF from surface normals stored on the CPU
   */
  void compute(const pcl::PointCloud<pcl::Normal>& normals);
  /*
   * compute MF from point cloud 
   */
  void compute(const pcl::PointCloud<pcl::PointXYZ>::Ptr & pc);

  MatrixXf mfAxes();
  const Matrix3f& cRmf() { return cRmf_;};
  const VectorXu& labels();
  cv::Mat labelsImg();
  cv::Mat normalsImg();
  cv::Mat smoothNormalsImg();
  cv::Mat smoothDepthImg();
  cv::Mat smoothDepth(){ return this->depthFilter_->getOutput();};
  cv::Mat normalsImgRaw(){ return normalExtract_->normalsImg();};
  cv::Mat overlaySeg(cv::Mat img);

  Matrix<uint8_t,Dynamic,Dynamic> mfAxCols_;

private:
    bool haveLabels_;

    double residual_, D_KL_;
    uint32_t nCGIter_;

    jsc::TimerLog tLog_;
    ofstream fout_;

    std::string mode_;
    cudaPcl::CfgSmoothNormals cfgNormals_;
    uint32_t w_, h_;

    mmf::OptSO3 * optSO3_;
    cudaPcl::DepthGuidedFilterGpu<float> *depthFilter_;
    cudaPcl::NormalExtractSimpleGpu<float> *normalExtract_;

    Matrix3f cRmf_;
    VectorXu z_;

    /*
     * runs the actual compute; assumes that normalExtract_ contains
     * the normals on GPU already.
     */
    void compute_();
};

// ------------------- impl --------------------------------------
RealtimeMF::RealtimeMF(std::string mode, const CfgOptSO3& cfgOptSO3,
    const cudaPcl::CfgSmoothNormals& cfgNormals)
  : haveLabels_(false), 
  residual_(0.0), D_KL_(0.0), nCGIter_(cfgOptSO3.nCGIter),
  tLog_("./timer.log",3,10,"TimerLog"),
  fout_("./stats.log",ofstream::out),
  mode_(mode), 
  cfgNormals_(cfgNormals),
  optSO3_(NULL),depthFilter_(NULL), normalExtract_(NULL),
    cRmf_(MatrixXf::Identity(3,3))
{
  mfAxCols_ = Matrix<uint8_t,Dynamic,Dynamic>(6,3);
  mfAxCols_ << 255,0,0,
              255,0,0,
              0,255,0,
              0,255,0,
              0,0,255,
              0,0,255;
  cout<<"inititalizing optSO3"<<endl
    <<"  mode = "<<mode_<<endl
    <<"  sigma = "<<cfgOptSO3.sigma<<endl
    <<"  tMax = "<<cfgOptSO3.tMax << " dt= "<<cfgOptSO3.dt<<endl
    <<"  nCGIter = "<<cfgOptSO3.nCGIter<<endl;

  if(mode_.compare("direct") == 0)
  {
    optSO3_ = new mmf::OptSO3(cfgOptSO3.sigma, 
        cfgOptSO3.tMax, cfgOptSO3.dt);
//    nCGIter_ = 10; // cannot do that many iterations
  }else if (mode_.compare("approx") == 0){
    optSO3_ = new mmf::OptSO3Approx(cfgOptSO3.sigma,
        cfgOptSO3.tMax, cfgOptSO3.dt);
//    nCGIter_ = 25;
  }else if (mode_.compare("vmf") == 0){
    optSO3_ = new mmf::OptSO3vMF(cfgOptSO3.sigma,
        cfgOptSO3.tMax, cfgOptSO3.dt);
//    nCGIter_ = 25;
  }
};

RealtimeMF::~RealtimeMF()
{
  delete optSO3_;
  delete normalExtract_;
  delete depthFilter_;
  fout_.close();
};

void RealtimeMF::compute(const uint16_t* depth, uint32_t w, uint32_t h)
{
  w_ = w; h_ = h;
  tLog_.tic(-1); // reset all timers
  if(!depthFilter_)
  {
    depthFilter_ = new cudaPcl::DepthGuidedFilterGpu<float>(w_,h_,
        cfgNormals_.eps,cfgNormals_.B);
    normalExtract_ = new
      cudaPcl::NormalExtractSimpleGpu<float>(cfgNormals_.f_d,
          w_,h_,cfgNormals_.compress);
  }
  cout<<" -- guided filter for depth image "<<w_<<" "<<h_<<endl;
  cv::Mat dMap(h_,w_,CV_16U,const_cast<uint16_t*>(depth));
  cout<<dMap.rows<<" "<<dMap.cols<<endl;
  depthFilter_->filter(dMap);
  tLog_.toctic(0,1);
  cout<<" -- extract surface normals on GPU"<<endl;
  normalExtract_->computeGpu(depthFilter_->getDepthDevicePtr(),w_,h_);
  compute_();
};

void RealtimeMF::compute(const pcl::PointCloud<pcl::Normal>& normals)
{
  // pcl::Normal is a float[4] array per point. the 4th entry is the
  // curvature
  w_ = normals.width; h_ = normals.height;
  tLog_.tic(-1); // reset all timers
  tLog_.toctic(0,1);
  if(!normalExtract_)
  {
    normalExtract_ = new
      cudaPcl::NormalExtractSimpleGpu<float>(cfgNormals_.f_d,
          w_,h_,cfgNormals_.compress);
  }
  normalExtract_->setNormalsCpu(normals);
  compute_();
}

void RealtimeMF::compute(const pcl::PointCloud<pcl::PointXYZ>::Ptr & pc)
{
  // pcl::Normal is a float[4] array per point. the 4th entry is the
  // curvature
  w_ = pc->width; h_ = pc->height;
  tLog_.tic(-1); // reset all timers
  tLog_.toctic(0,1);
  if(!normalExtract_)
  {
    normalExtract_ = new
      cudaPcl::NormalExtractSimpleGpu<float>(cfgNormals_.f_d,
          w_,h_,cfgNormals_.compress);
  }
  normalExtract_->compute(pc);
  compute_();
};


void RealtimeMF::compute_()
{
  // get compressed normals
  tLog_.toctic(1,2);
  int32_t nComp = 0;
  float* d_nComp = normalExtract_->d_normalsComp(nComp);
  cout<<" -- compressed to "<<nComp<<" normals"<<endl;
  // optimize normals
  tLog_.toc(2); // total time
  optSO3_->updateExternalGpuNormals(d_nComp,nComp,3,0);
  residual_ = optSO3_->conjugateGradientCUDA(cRmf_,nCGIter_);
  D_KL_ = optSO3_->D_KL_axisUnif();
  cout<<" -- optimized rotation D_KL to unif "<<D_KL_<<endl
    <<cRmf_<<endl;
  tLog_.setDt(3,optSO3_->dtPrep());
  tLog_.setDt(4,optSO3_->dtCG());
  tLog_.logCycle();
  haveLabels_ = false;
};


MatrixXf RealtimeMF::mfAxes()
{
  MatrixXf mfAx = MatrixXf::Zero(3,6);
  for(uint32_t k=0; k<6; ++k){
    int j = k/2; // which of the rotation columns does this belong to
    float sign = (- float(k%2) +0.5f)*2.0f; // sign of the axis
    mfAx.col(k) = sign*cRmf_.col(j);
  }
  return mfAx;
}


const VectorXu& RealtimeMF::labels()
{
  if(!haveLabels_)
  {
    if(z_.rows() != w_*h_) z_.resize(w_*h_);
    normalExtract_->uncompressCpu(optSO3_->z().data(),
        optSO3_->z().rows(), z_.data(), z_.rows());
    haveLabels_ = true;
  }
  return z_;
};

cv::Mat RealtimeMF::labelsImg()
{
  labels();
  cv::Mat zIrgb(h_,w_,CV_8UC3);
  for(uint32_t i=0; i<w_; i+=1)
    for(uint32_t j=0; j<h_; j+=1)
      if(z_(w_*j +i) < 6) 
//      if(this->normalsImg_.at<cv::Vec3f>(j,i)[0] ==
//          this->normalsImg_.at<cv::Vec3f>(j,i)[0])
      {
        uint32_t idz = z_(w_*j +i);
        zIrgb.at<cv::Vec3b>(j,i)[0] = mfAxCols_(5-idz,0);
        zIrgb.at<cv::Vec3b>(j,i)[1] = mfAxCols_(5-idz,1);
        zIrgb.at<cv::Vec3b>(j,i)[2] = mfAxCols_(5-idz,2);
      }else{
        zIrgb.at<cv::Vec3b>(j,i)[0] = 255;
        zIrgb.at<cv::Vec3b>(j,i)[1] = 255;
        zIrgb.at<cv::Vec3b>(j,i)[2] = 255;
      }
  return zIrgb;
};

cv::Mat RealtimeMF::normalsImg()
{
  cv::Mat nI(h_,w_,CV_8UC3);
  cv::Mat nIRGB(h_,w_,CV_8UC3);
  this->normalsImgRaw().convertTo(nI,CV_8UC3,127.5f,127.5f);
  cv::cvtColor(nI,nIRGB,CV_RGB2BGR);
  return nIRGB;
}

cv::Mat RealtimeMF::smoothDepthImg()
{
  cv::Mat dI(h_,w_,CV_8UC1);
  this->smoothDepth().convertTo(dI,CV_8UC1,255./4.,-1.9);
  return dI;
}

cv::Mat RealtimeMF::overlaySeg(cv::Mat img)
{
  cv::Mat rgb;
  if(img.channels() == 1)
  {
    std::vector<cv::Mat> grays(3);
    grays.at(0) = img;
    grays.at(1) = img;
    grays.at(2) = img;
    cv::merge(grays, rgb);
  }else{
    rgb = img;
  }
  cv::Mat zI = labelsImg();
  cv::Mat Iout;
  cv::addWeighted(rgb , 0.7, zI, 0.3, 0.0, Iout);
  projectDirections(Iout,mfAxes(),cfgNormals_.f_d,mfAxCols_);
  return Iout;
};
