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
    virtual void visualizeNormals();

   shared_ptr<RealtimeMF> pRtmf_;
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
  this->update();
//  cout<<pRtmf_->cRmf()<<endl;
}

void RealtimeMF_openni::visualizeNormals()
{
  cout<<"visualize Normals"<<endl;
  cv::Mat Iseg = pRtmf_->overlaySeg(this->rgb_);
  cv::imshow("seg",Iseg);
}

#endif
