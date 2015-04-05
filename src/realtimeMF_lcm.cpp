/* Copyright (c) 2015, Julian Straub <jstraub@csail.mit.edu> Licensed
 * under the MIT license. See the license file LICENSE.
 */


#include <iostream>
#include <fstream>
#include <string>

#include <lcmtypes/kinect_depth_msg_t.h>
#include <lcmtypes/kinect_frame_msg_t.h>

#include <lcm/lcm-cpp.hpp>

#include "lcmtypes/bot_core.hpp"
#include "lcmtypes/kinect.hpp"

#include <rtmf/rtmf.hpp>

class RealtimeMF_lcm
{
  public:
    RealtimeMF_lcm(shared_ptr<lcm::LCM>& lcm, 
        shared_ptr<RealtimeMF>& pRtmf);
    virtual ~RealtimeMF_lcm();

    virtual void rgbd_cb(const lcm::ReceiveBuffer* rbuf, const
        std::string& channel, const  kinect::frame_msg_t* msg);

  protected:
    virtual void visualizeNormals();

   shared_ptr<lcm::LCM> lcm_;
   shared_ptr<RealtimeMF> pRtmf_;

   cv::Mat rgb_;

};


RealtimeMF_lcm::RealtimeMF_lcm(shared_ptr<lcm::LCM>& lcm,
    shared_ptr<RealtimeMF>& pRtmf)
  : lcm_(lcm), pRtmf_(pRtmf)
{
  lcm_->subscribe( "KINECT_FRAME",&RealtimeMF_lcm::rgbd_cb,this);
};

RealtimeMF_lcm::~RealtimeMF_lcm()
{
}

void RealtimeMF_lcm::depth_cb(const lcm::ReceiveBuffer* rbuf, const
    std::string& channel, const  kinect::frame_msg_t* msg)
{
  cout<<"rgbd @"<<msg->image.timestamp<<endl;
  std::cout << "depth data fromat: " << msg->depth.depth_data_format << std::endl;
  std::cout << "image data fromat: " << msg->image.image_data_format << std::endl;

  if (msg->image.image_data_format == kinect::image_msg_t::VIDEO_RGB){
    rgb_ = cv::Mat(msg->image.height, msg->image.width, cv::U8C3,
        msg->image.image_data);
  }else{
    std::cout << "Format not recognized: " << msg->image.image_data_format << std::endl;
  }

  if (msg->depth.depth_data_format == kinect::depth_msg_t::DEPTH_11BIT){
//    rgb_ = cv::Mat(msg->depth.height, msg->depth.width, cv::U8C3,
//        msg->depth.image_data);
  }else{
    std::cout << "Format not recognized: " << msg->depth.depth_data_format << std::endl;
  }

//  pRtmf_->compute(depth,w,h);
//  this->update();
//  cout<<pRtmf_->cRmf()<<endl;
}

void RealtimeMF_lcm::visualizeNormals()
{
  cout<<"visualize Normals"<<endl;
  cv::Mat Iseg = pRtmf_->overlaySeg(this->rgb_);
  cv::imshow("seg",Iseg);
}

#include <boost/program_options.hpp>
namespace po = boost::program_options;



