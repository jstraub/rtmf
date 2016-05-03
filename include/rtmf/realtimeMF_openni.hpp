/* Copyright (c) 2015, Julian Straub <jstraub@csail.mit.edu> Licensed
 * under the MIT license. See the license file LICENSE.
 */
#ifndef REALTIME_MF_OPENNI_HPP_INCLUDED
#define REALTIME_MF_OPENNI_HPP_INCLUDED

#include <string>

#include <pcl/io/openni_grabber.h>
#include <pcl/io/openni_camera/openni_depth_image.h>
#include <pcl/io/openni_camera/openni_image.h>

#include <Eigen/Dense>

#include <jsCore/timerLog.hpp>

#include <mmf/optimizationSO3.hpp>
#include <mmf/optimizationSO3_approx.hpp>
#include <mmf/optimizationSO3_vmf.hpp>

#include <cudaPcl/openniSmoothNormalsGpu.hpp>

#include <rtmf/rtmf.hpp>

using namespace Eigen;
using namespace std;

class RealtimeMF_openni : public cudaPcl::OpenniVisualizer
{
  public:
    RealtimeMF_openni(shared_ptr<RealtimeMF>& pRtmf);
    virtual ~RealtimeMF_openni();

    virtual void depth_cb(const uint16_t* depth, uint32_t
        w, uint32_t h);

  protected:
    virtual void visualizeRGB();

   shared_ptr<RealtimeMF> pRtmf_;
   cv::Mat normalsImg_;

//    void visualizePC()
//    {
//      if (normalsImg_.empty() || normalsImg_.rows == 0 || normalsImg_.cols
//          == 0) return;
////      cv::Mat nI (normalsImg_.rows,normalsImg_.cols, CV_8UC3);
////      //  cv::Mat nIRGB(normalsImg_.rows,normalsImg_.cols,CV_8UC3);
////      normalsImg_.convertTo(nI, CV_8UC3, 127.5,127.5);
////      cv::cvtColor(nI,nIRGB_,CV_RGB2BGR);
////      cv::imshow("normals",nIRGB_);
////      if (compress_)  cv::imshow("dcomp",normalsComp_);
////
////      if (false) {
////        // show additional diagnostics
////        std::vector<cv::Mat> nChans(3);
////        cv::split(normalsImg_,nChans);
////        cv::Mat nNans = nChans[0].clone();
////        showNans(nNans);
////        cv::imshow("normal Nans",nNans);
////        cv::Mat haveData = normalExtract->haveData();
////        cv::imshow("haveData",haveData*200);
////      }
//
//#ifdef USE_PCL_VIEWER
//      pc_ = pcl::PointCloud<pcl::PointXYZRGB>::Ptr(new
//          pcl::PointCloud<pcl::PointXYZRGB>());
//      for (uint32_t i=0; i<normalsImg_.rows; ++i) 
//        for (uint32_t j=0; j<normalsImg_.cols; ++j) {
//          pcl::PointXYZRGB p;
//          p.x = normalsImg_.at<cv::Vec3f>(i,j)[0];
//          p.y = normalsImg_.at<cv::Vec3f>(i,j)[1];
//          p.z = normalsImg_.at<cv::Vec3f>(i,j)[2];
//          p.rgb = 0;
//          float norm = p.x*p.x+p.y*p.y+p.z*p.z;
//          if (0.98 <= norm && norm <= 1.02) this->pc_->push_back(p);
//        }
//      if (this->pc_->size() > 0) {
//        if(!this->viewer_->updatePointCloud(this->pc_, "pc"))
//          this->viewer_->addPointCloud(this->pc_, "pc");
//      }
//#endif
//    }
};

// -------------------------------- impl -----------------------------------

RealtimeMF_openni::RealtimeMF_openni(shared_ptr<RealtimeMF>& pRtmf)
  : pRtmf_(pRtmf)
{};

RealtimeMF_openni::~RealtimeMF_openni()
{
}

void RealtimeMF_openni::depth_cb(const uint16_t* depth, uint32_t w,
    uint32_t h)
{
  pRtmf_->compute(depth,w,h);
//  normalsImg_ = pRtmf_->normalsImg_;
//  cout<<pRtmf_->cRmf()<<endl;
  cv::Mat Iseg;
  cout<<"visualize Normals"<<endl;
  {
    //boost::mutex::scoped_lock updateLock(this->updateModelMutex);
    Iseg = pRtmf_->overlaySeg(this->rgb_);
  }
  cv::imshow("seg",Iseg);
  this->update();
}

void RealtimeMF_openni::visualizeRGB()
{
//  cv::Mat Iseg;
//  cout<<"visualize Normals"<<endl;
//  {
//    //boost::mutex::scoped_lock updateLock(this->updateModelMutex);
//    Iseg = pRtmf_->overlaySeg(this->rgb_);
//  }
//  cv::imshow("seg",Iseg);
}

#endif
