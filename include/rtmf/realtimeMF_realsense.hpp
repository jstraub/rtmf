/* Copyright (c) 2015, Julian Straub <jstraub@csail.mit.edu> Licensed
 * under the MIT license. See the license file LICENSE.
 */
#ifndef REALTIME_MF_realsense_HPP_INCLUDED
#define REALTIME_MF_realsense_HPP_INCLUDED

#include <string>

#include <Eigen/Dense>

#include <jsCore/timerLog.hpp>

#include <mmf/optimizationSO3.hpp>
#include <mmf/optimizationSO3_approx.hpp>
#include <mmf/optimizationSO3_vmf.hpp>

#include <cudaPcl/openniSmoothNormalsGpu.hpp>
#include <rgbdGrabber/realSenseGrabber.hpp>

#include <rtmf/rtmf.hpp>

using namespace Eigen;
using namespace std;

class RealtimeMF_realsense : public rgbdGrabber::RealSenseGrabber
{
  public:
    RealtimeMF_realsense(shared_ptr<RealtimeMF>& pRtmf, uint32_t w, uint32_t h, uint32_t fps);
    virtual ~RealtimeMF_realsense();

    virtual void rgbd_cb(const uint8_t* rgb, const uint16_t* depth);

  protected:

   shared_ptr<RealtimeMF> pRtmf_;
   cv::Mat normalsImg_;
   cv::Mat rgb_;

};

// -------------------------------- impl -----------------------------------

RealtimeMF_realsense::RealtimeMF_realsense(shared_ptr<RealtimeMF>& pRtmf,
  uint32_t w, uint32_t h, uint32_t fps)
  : rgbdGrabber::RealSenseGrabber(w,h,fps), pRtmf_(pRtmf)
{};

RealtimeMF_realsense::~RealtimeMF_realsense()
{
}

void RealtimeMF_realsense::rgbd_cb(const uint8_t* rgb, const uint16_t* depth)
{
  rgb_ = cv::Mat(h_,w_,CV_8UC3,const_cast<uint8_t*>(rgb));
  pRtmf_->compute(depth,w_,h_);
//  normalsImg_ = pRtmf_->normalsImg_;
//  cout<<pRtmf_->cRmf()<<endl;
  cv::Mat Iseg;
  cout<<"visualize Normals"<<endl;
  {
    Iseg = pRtmf_->overlaySeg(rgb_);
  }
  cv::imshow("seg",Iseg);
  cv::waitKey(1);
}

#endif
