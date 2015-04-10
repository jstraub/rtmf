/* Copyright (c) 2015, Julian Straub <jstraub@csail.mit.edu> Licensed
 * under the MIT license. See the license file LICENSE.
 */
#pragma once

#include <zlib.h>
#include <glib.h>

#include <lcm/lcm-cpp.hpp>

#include "lcmtypes/kinect.hpp"
#include <lcmtypes/kinect_depth_msg_t.h>
#include <lcmtypes/kinect_frame_msg_t.h>

//#include <bot_param/param_client.h>
#include <lcmtypes/bot_core/image_t.hpp>
#include <lcmtypes/bot_core/images_t.hpp>

#include <rtmf/rtmf.hpp>
#include <rtmf/jpeg-utils-ijg.h>

class RealtimeMF_lcm
{
  public:
    RealtimeMF_lcm(shared_ptr<lcm::LCM>& lcm, 
        shared_ptr<RealtimeMF>& pRtmf, 
        string channel="KINECT_FRAME",
        string outPath="");
    virtual ~RealtimeMF_lcm();

    virtual void rgbd_cb(const lcm::ReceiveBuffer* rbuf, const
        string& channel, const  kinect::frame_msg_t* msg);
    virtual void disparity_cb(const lcm::ReceiveBuffer* rbuf, const
        string& channel, const  bot_core::images_t* msg);

  protected:
    virtual void visualizeNormals();

   shared_ptr<lcm::LCM> lcm_;
   shared_ptr<RealtimeMF> pRtmf_;
   string outPath_;

   cv::Mat rgb_;
   cv::Mat d_;

};


RealtimeMF_lcm::RealtimeMF_lcm(shared_ptr<lcm::LCM>& lcm,
    shared_ptr<RealtimeMF>& pRtmf, string outPath)
  : lcm_(lcm), pRtmf_(pRtmf), outPath_(outPath)
{
  if(channel.compre("KINECT_FRAME") == 0)
    lcm_->subscribe( "KINECT_FRAME",&RealtimeMF_lcm::rgbd_cb,this);
  else if(channel.compre("CAMERA") == 0)
    lcm_->subscribe( "CAMERA",&RealtimeMF_lcm::disparity_cb,this);
  else
    cout<<"channel "<<channel<<" unkown!!"<<endl;
};

RealtimeMF_lcm::~RealtimeMF_lcm()
{}

void RealtimeMF_lcm::rgbd_cb(const lcm::ReceiveBuffer* rbuf, const
    string& channel, const  kinect::frame_msg_t* msg)
{
  cout<<"rgbd @"<<msg->image.timestamp<<endl;
  cout << "depth data fromat: " << int(msg->depth.depth_data_format)
    << endl;
  cout << "image data fromat: " << int(msg->image.image_data_format) 
    << endl;

  if (msg->image.image_data_format == kinect::image_msg_t::VIDEO_RGB)
  {
    cout << "raw rgb image "<<endl;
    rgb_ = cv::Mat(msg->image.height, msg->image.width, CV_8UC3);
//        const_cast<uint8_t*>(&(msg->image.image_data[0])));
    memcpy(rgb_.data, &(msg->image.image_data[0]), msg->image.width *
        msg->image.height * 3);
  }else if (msg->image.image_data_format == kinect::image_msg_t::VIDEO_RGB_JPEG){
    cout << "jpg rgb image "<<endl;
    rgb_ = cv::Mat(msg->image.height, msg->image.width, CV_8UC3);
    jpegijg_decompress_8u_rgb (&(msg->image.image_data[0]),
          msg->image.image_data_nbytes, rgb_.data, msg->image.width,
          msg->image.height, msg->image.width * 3);
  }else{
    cout << "Format not recognized: " << msg->image.image_data_format << endl;
    rgb_ = cv::Mat::zeros(msg->image.height, msg->image.width, CV_8UC3);
  }

  if(msg->depth.compression == KINECT_DEPTH_MSG_T_COMPRESSION_NONE)
//  if (msg->depth.depth_data_format == kinect::depth_msg_t::DEPTH_MM)
  {
    cout << "MM depth image"<<msg->depth.height<<"x"<< msg->depth.width<<endl;
    d_ = cv::Mat(msg->depth.height, msg->depth.width, CV_16SC1,
        const_cast<uint8_t*>(&(msg->depth.depth_data[0])));
    cv::imshow("depth",d_);
  }else{
    cout << " depth compressed: " << msg->depth.depth_data_format
      <<msg->depth.height<<"x"<< msg->depth.width<<endl;
    unsigned long dlen = msg->depth.uncompressed_size;
    d_ = cv::Mat(msg->depth.height, msg->depth.width, CV_16SC1);
    uncompress(d_.data, &dlen, &(msg->depth.depth_data[0]),
        msg->depth.depth_data_nbytes);
  }

  pRtmf_->compute(d_);
//  this->update();
//  cout<<pRtmf_->cRmf()<<endl;
//
 visualizeNormals(); cv::waitKey(10);
}

void RealtimeMF_lcm::disparity_cb(const lcm::ReceiveBuffer* rbuf, const
    string& channel, const  bot_core::images_t* msg)
{
  cout<<"multisense @"<<msg->images.utime<<" #imgs "<<msg->n_images<<endl;

  if (msg->images[0].pixelformat == BOT_CORE_IMAGE_T_PIXEL_FORMAT_RGB ){
    rgb_buf_ = msg->images[0].data;
    rgb_ = cv::Mat(msg->images[0].height, msg->images[0].width, CV_8UC3);
    memcpy(rgb_.data, &(msg->images[0].image_data[0]),
        msg->images[0].width * msg->images[0].height * 3);
  }else if (msg->images[0].pixelformat == BOT_CORE_IMAGE_T_PIXEL_FORMAT_GRAY ){
    cv::Mat gray = cv::Mat(msg->images[0].height, msg->images[0].width,
        CV_8UC1);
    memcpy(gray.data, &(msg->images[0].image_data[0]),
        msg->images[0].width * msg->images[0].height);
  }else if (msg->images[0].pixelformat == BOT_CORE_IMAGE_T_PIXEL_FORMAT_MJPEG ){
    jpegijg_decompress_8u_rgb (&(msg->images[0].image_data[0]),
          msg->images[0].size, rgb_.data, msg->images[0].width,
          msg->images[0].height, msg->image[0].width * 3);
  }else{
    std::cout << "multisense_utils::unpack_multisense | type not understood\n";
    exit(-1);
  }
//
//  // TODO: support other modes (as in the renderer)
//  if (msg->image_types[1] == BOT_CORE_IMAGES_T_DISPARITY_ZIPPED ) {
//    unsigned long dlen = msg->images[0].width*msg->images[0].height*2 ;//msg->depth.uncompressed_size;
//    d_ = cv::Mat(msg->images[1].height, msg->images[1].width, CV_16SC1);
//    uncompress(d_.data , &dlen, msg->images[1].data, msg->images[1].size);
//  } else{
//    std::cout << "multisense_utils::unpack_multisense | depth type not understood\n";
//    exit(-1);
//  }  
//
//    // Remove disconnect components. TODO: if needed this can also be
//    // used for the depth data
//    if (size_threshold_ > 0){
//      // Distance threshold conversion:
//      float k00 = 1/repro_matrix(2,3);
//      float baseline = 1/repro_matrix(3,2);
//      float mDisparityFactor = 1/k00/baseline;
//      float thresh = 16.0/mDisparityFactor/depth_threshold_;
//      miu_.removeSmall(disparity_orig_temp, thresh, size_threshold_);
//    }
//
//    //std::copy(msg->images[1].data.data()             , msg->images[1].data.data() + (msg->images[1].size) ,
//    //          disparity_orig_temp.data);
//
//    // disparity_orig_temp.data = msg->images[1].data.data();   // ... is a simple assignment possible?
//    cv::Mat_<float> disparity_orig(h, w);
//    disparity_orig = disparity_orig_temp;
//
//    disparity_buf_.resize(h * w);
//    cv::Mat_<float> disparity(h, w, &(disparity_buf_[0]));
//    disparity = disparity_orig / 16.0;
//
//    // Allocate buffer for reprojection output
//    points_buf_.resize(h * w);
//    cv::Mat_<cv::Vec3f> points(h, w, &(points_buf_[0]));
//
//    // Do the reprojection in open space
//    static const bool handle_missing_values = true;
//    cv::reprojectImageTo3D(disparity, points, repro_matrix, handle_missing_values);
//
////  pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud(
////      new pcl::PointCloud<pcl::PointXYZRGB>);
////  ms_utils_->unpack_multisense(msg,Q_,cloud);  
//
//  pRtmf_->compute(d_);
////  this->update();
////  cout<<pRtmf_->cRmf()<<endl;
////
// visualizeNormals(); cv::waitKey(10);
}

void RealtimeMF_lcm::visualizeNormals()
{
  static int frameNr = 0;
  cout<<"visualize Normals"<<endl;
  cv::Mat Iseg = pRtmf_->overlaySeg(this->rgb_);
  cv::imshow("seg",Iseg);
  if (outPath_.compare("") != 0)
  {
    stringstream ss;
    ss << outPath_ << "./frame"
      << std::setw(7) << std::setfill('0') << frameNr<<".png";
    cout<<"writing to "<<ss.str()<<endl;
    cv::imwrite(ss.str().data(),Iseg);
  }
  ++ frameNr;
}

