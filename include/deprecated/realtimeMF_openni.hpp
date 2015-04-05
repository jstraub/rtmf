#ifndef REALTIME_MF_OPENNI_HPP_INCLUDED
#define REALTIME_MF_OPENNI_HPP_INCLUDED

#include <string>

#include <pcl/io/openni_grabber.h>
#include <pcl/io/openni_camera/openni_depth_image.h>
#include <pcl/io/openni_camera/openni_image.h>

#include <Eigen/Dense>

#include <mmf/optimizationSO3.hpp>
#include <mmf/optimizationSO3_approx.hpp>
#include <mmf/optimizationSO3_vmf.hpp>

#include <cudaPcl/openniSmoothNormalsGpu.hpp>
#include <cudaPcl/timerLog.hpp>
#include <cudaPcl/gpuMatrix.hpp>

using namespace Eigen;
using namespace std;

class RealtimeMF_openni : public cudaPcl::OpenniSmoothNormalsGpu
{
  public:
    RealtimeMF_openni(std::string mode, uint32_t w, uint32_t h, 
        float f_d);
    virtual ~RealtimeMF_openni();

    virtual void normals_cb(float* d_normals, uint8_t* haveData, uint32_t
        w, uint32_t h);

    virtual void visualizeNormals();

    void init(int w, int h);

    static void projectDirections(cv::Mat& I, const MatrixXf& dirs, double f_d,
        const Matrix<uint8_t, Dynamic,Dynamic>& colors);

  protected:

    double residual_, D_KL_;
    uint32_t nCGIter_;

    cudaPcl::TimerLog tLog_;

    uint32_t w_, h_;

    ofstream fout_;
    std::string mode_;

    Matrix3f kRw_;
    MatrixXf mfAxes_;

    mmf::OptSO3 *optSO3_;

    VectorXu z_;
    cv::Mat zIrgb_;
    cv::Mat Icomb_;

    Matrix3d kRv;
    Vector3d dt0;
};

// -------------------------------- impl -----------------------------------

RealtimeMF_openni::RealtimeMF_openni(std::string mode, uint32_t w, uint32_t h,
    float f_d)
  : OpenniSmoothNormalsGpu(f_d, 0.2*0.2, 9, true), 
  // TODO check B and eps!
  residual_(0.0), D_KL_(0.0), nCGIter_(10),
  tLog_("./timer.log",3,10,"TimerLog"),
  w_(w), h_(h),
  fout_("./stats.log",ofstream::out),
  mode_(mode), kRw_(Matrix3f::Identity()),
  optSO3_(NULL)
{};

RealtimeMF_openni::~RealtimeMF_openni()
{
  delete optSO3_;
  fout_.close();
}

void RealtimeMF_openni::init(int w, int h)
{
  if(optSO3_ != NULL) return;
  cout<<"inititalizing optSO3"<<endl;
  // TODO: use weights
  // TODO: make sure N is float all over the place
  //optSO3_ = new OptSO3(25.0f*M_PI/180.0f,d_n,w,h,NULL);//,d_weights);
  if(mode_.compare("direct") == 0)
  {
#ifndef WEIGHTED
    optSO3_ = new mmf::OptSO3(5.0f*M_PI/180.0f);//,d_weights);
#else
    optSO3_ = new mmf::OptSO3(5.0f*M_PI/180.0f);
//        normalExtractor_.d_weights());
#endif
    nCGIter_ = 10; // cannot do that many iterations
  }else if (mode_.compare("approx") == 0){
#ifndef WEIGHTED
    optSO3_ = new mmf::OptSO3Approx(5.0f*M_PI/180.0f);//,d_weights);
#else
    optSO3_ = new mmf::OptSO3Approx(5.0f*M_PI/180.0f);
//        normalExtractor_.d_weights());
#endif
    nCGIter_ = 25;
  }else if (mode_.compare("vmf") == 0){
#ifndef WEIGHTED
    optSO3_ = new mmf::OptSO3vMF(5.0f*M_PI/180.0f);//,d_weights);
#else
    optSO3_ = new mmf::OptSO3vMF(5.0f*M_PI/180.0f);
//        normalExtractor_.d_weights());
#endif
    nCGIter_ = 25;
  }
}

void RealtimeMF_openni::projectDirections(cv::Mat& I, const MatrixXf& dirs,
    double f_d, const Matrix<uint8_t,Dynamic,Dynamic>& colors)
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
}


void RealtimeMF_openni::normals_cb(float* d_normals, uint8_t* haveData,
    uint32_t w, uint32_t h)
{
  tLog_.tic(-1); // reset all timers
  int32_t nComp = 0;
  float* d_nComp = this->normalExtract->d_normalsComp(nComp);
  Matrix3f kRwBefore = kRw_;
  tLog_.toctic(0,1);

  optSO3_->updateExternalGpuNormals(d_nComp,nComp,3,0);
  double residual = optSO3_->conjugateGradientCUDA(kRw_,nCGIter_);
  double D_KL = optSO3_->D_KL_axisUnif();
  tLog_.toctic(1,2);

  {
    boost::mutex::scoped_lock updateLock(this->updateModelMutex);
    this->normalsImg_ = this->normalExtract->normalsImg();
    if(z_.rows() != w*h) z_.resize(w*h);
    this->normalExtract->uncompressCpu(optSO3_->z().data(), optSO3_->z().rows(),
        z_.data(), z_.rows());

    mfAxes_ = MatrixXf::Zero(3,6);
    for(uint32_t k=0; k<6; ++k){
      int j = k/2; // which of the rotation columns does this belong to
      float sign = (- float(k%2) +0.5f)*2.0f; // sign of the axis
      mfAxes_.col(k) = sign*kRw_.col(j);
    }
    D_KL_= D_KL;
    residual_ = residual;
    this->update_ = true;
    updateLock.unlock();
  }

  tLog_.toc(2); // total time
  tLog_.logCycle();
  cout<<"delta rotation kRw_ = \n"<<kRwBefore*kRw_.transpose()<<endl;
  cout<<"---------------------------------------------------------------------------"<<endl;
  tLog_.printStats();
  cout<<" residual="<<residual_<<"\t D_KL="<<D_KL_<<endl;
  cout<<"---------------------------------------------------------------------------"<<endl;
  fout_<<D_KL_<<" "<<residual_<<endl; fout_.flush();

////  return kRw_;
//  {
//    boost::mutex::scoped_lock updateLock(this->updateModelMutex);
////    pcl::PointCloud<pcl::PointXYZRGB>::ConstPtr nDispPtr =
////      normalExtract->normalsPc();
////    nDisp_ = pcl::PointCloud<pcl::PointXYZRGB>::Ptr( new
////        pcl::PointCloud<pcl::PointXYZRGB>(*nDispPtr));
////    this->normalsImg_ = this->normalExtract->normalsImg();
//    this->update_ = true;
//  }
};

void RealtimeMF_openni::visualizeNormals()
{
  if (normalsImg_.empty() || normalsImg_.rows == 0 || normalsImg_.cols == 0) return;
  cout<<"entering RealtimeMF_openni::visualizeNormals"<<endl;
//  pcl::PointCloud<pcl::PointXYZRGB>::Ptr nDisp(
//      new pcl::PointCloud<pcl::PointXYZRGB>(*nDisp_));
  const w = this->normalsImg_.cols;
  const h = this->normalsImg_.rows;

  cv::Mat nI(h,w,CV_8UC3);
  cv::Mat nIRGB(h,w,CV_8UC3);
  this->normalsImg_.convertTo(nI,CV_8UC3,127.5f,127.5f);
  cv::cvtColor(nI,this->nIRGB_,CV_RGB2BGR);
  cv::imshow("normals",this->nIRGB_);

//  uint32_t k=0;

  Matrix<uint8_t,Dynamic,Dynamic> mfAxCols(6,3);
  mfAxCols << 255,0,0,
              255,0,0,
              0,255,0,
              0,255,0,
              0,0,255,
              0,0,255;

//  cout<<" z shape "<<z_.rows()<<" "<< w<<" " <<h<<endl;
//  cv::Mat Iz(nDisp->height/SUBSAMPLE_STEP,nDisp->width/SUBSAMPLE_STEP,CV_8UC1);
  zIrgb_ = cv::Mat(h,w,CV_8UC3);
//  for(uint32_t i=0; i<w; i+=1)
//    for(uint32_t j=0; j<h; j+=1)
//      if(this->normalsImg_.at<cv::Vec3f>(j,i)[0]==this->normalsImg_.at<cv::Vec3f>(j,i)[0])
//      {
//        uint32_t idz = z_(w*j +i);
//        //            cout<<"k "<<k<<" "<< z_.rows() <<"\t"<<z_(k)<<"\t"<<int32_t(idz)<<endl;
//        cout<<j<<" "<<i<<" id="<<idz<<endl;
//        zIrgb_.at<cv::Vec3b>(j,i)[0] = mfAxCols(idz,0);
//        zIrgb_.at<cv::Vec3b>(j,i)[1] = mfAxCols(idz,1);
//        zIrgb_.at<cv::Vec3b>(j,i)[2] = mfAxCols(idz,2);
////        k++;
//      }else{
//        zIrgb_.at<cv::Vec3b>(j,i)[0] = 255;
//        zIrgb_.at<cv::Vec3b>(j,i)[1] = 255;
//        zIrgb_.at<cv::Vec3b>(j,i)[2] = 255;
//      }

//  cout<<this->rgb_.rows <<" " << this->rgb_.cols<<endl;
  if(this->rgb_.rows>1 && this->rgb_.cols >1)
  {
    cv::addWeighted(this->rgb_ , 0.7, zIrgb, 0.3, 0.0, Icomb_);
    projectDirections(Icomb_,mfAxes_,this->f_d_,mfAxCols);
    cv::imshow("dbg",Icomb);
  }else{
    projectDirections(zIrgb_,mfAxes_,this->f_d_,mfAxCols);
    cv::imshow("dbg",zIrgb);
  }

};

#endif
