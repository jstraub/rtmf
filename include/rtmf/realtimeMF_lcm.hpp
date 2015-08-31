/* Copyright (c) 2015, Julian Straub <jstraub@csail.mit.edu> Licensed
 * under the MIT license. See the license file LICENSE.
 */
#pragma once

#include <zlib.h>
#include <glib.h>

#include <opencv2/opencv.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <lcm/lcm-cpp.hpp>
#include "lcmtypes/kinect.hpp"
#include <lcmtypes/kinect_depth_msg_t.h>
#include <lcmtypes/kinect_frame_msg_t.h>
//#include <bot_param/param_client.h>
#include <lcmtypes/bot_core/image_t.hpp>
#include <lcmtypes/bot_core/images_t.hpp>

#include <multisense_image_utils/multisense_image_utils.hpp>

#include <rtmf/rtmf.hpp>
#include <rtmf/jpeg-utils-ijg.h>
#include <cudaPcl/cv_helpers.hpp>

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

    multisense_image_utils miu_;
};


RealtimeMF_lcm::RealtimeMF_lcm(shared_ptr<lcm::LCM>& lcm,
    shared_ptr<RealtimeMF>& pRtmf, 
    string channel,
    string outPath)
  : lcm_(lcm), pRtmf_(pRtmf), outPath_(outPath)
{
  if(channel.compare("KINECT_FRAME") == 0)
    lcm_->subscribe( "KINECT_FRAME",&RealtimeMF_lcm::rgbd_cb,this);
  else if(channel.compare("CAMERA") == 0)
    lcm_->subscribe( "CAMERA",&RealtimeMF_lcm::disparity_cb,this);
  else
    cout<<"channel "<<channel<<" unkown!!"<<endl;
  cout<<"connected to channel "<<channel<<endl;
  
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
  cout<<"multisense @"<<" #imgs "<<msg->n_images<<endl;

  if (msg->images[0].pixelformat == bot_core::image_t::PIXEL_FORMAT_RGB ){
    cout<<"rgb image"<<endl;
//    rgb_buf_ = msg->images[0].data;
    rgb_ = cv::Mat(msg->images[0].height, msg->images[0].width, CV_8UC3);
    memcpy(rgb_.data, &(msg->images[0].data[0]),
        msg->images[0].width * msg->images[0].height * 3);
  }else if (msg->images[0].pixelformat == bot_core::image_t::PIXEL_FORMAT_GRAY ){
    cout<<"gray image"<<endl;
    cv::Mat gray = cv::Mat(msg->images[0].height, msg->images[0].width,
        CV_8UC1);
    memcpy(gray.data, &(msg->images[0].data[0]),
        msg->images[0].width * msg->images[0].height);
  }else if (msg->images[0].pixelformat == bot_core::image_t::PIXEL_FORMAT_MJPEG ){
    cout<<"jpeg image "<<msg->images[0].size<<" "<< msg->images[0].width <<" "<<msg->images[0].height<<endl;
    rgb_ = cv::Mat(msg->images[0].height, msg->images[0].width, CV_8UC3);
    jpegijg_decompress_8u_rgb ( &(msg->images[0].data[0]),
          msg->images[0].size, rgb_.data, msg->images[0].width,
          msg->images[0].height, msg->images[0].width * 3);
  }else{
    std::cout << "multisense_utils::unpack_multisense | type not understood\n";
    exit(-1);
  }
  cv::imshow("rgb",rgb_);

  // TODO: support other modes (as in the renderer)
  if (msg->image_types[1] == bot_core::images_t::DISPARITY_ZIPPED ) {
    cout<<"zipped depth"<<endl;
    unsigned long dlen = msg->images[1].width*msg->images[1].height*2 ;//msg->depth.uncompressed_size;
    d_ = cv::Mat(msg->images[1].height, msg->images[1].width, CV_16SC1);
    uncompress(d_.data , &dlen, &(msg->images[1].data[0]), msg->images[1].size);
  } else{
    std::cout << "multisense_utils::unpack_multisense | depth type not understood\n";
    exit(-1);
  }  
//  cv::imshow("d",d_);

  uint32_t h = msg->images[1].height;
  uint32_t w = msg->images[1].width;

  float decimate_ =32.0;
  float size_threshold_ = 1000; // in pixels
  float depth_threshold_ = 1000.0; // in m

  float repro[16] =  {1, 0, 0, -512.5, 0, 1, 0, -272.5, 0, 0, 0, 606.034, 0, 0, 14.2914745276283, 0};
  cv::Mat_<float> repro_matrix(4,4,repro);

    // Remove disconnect components. TODO: if needed this can also be
    // used for the depth data
    if (size_threshold_ > 0){
      // Distance threshold conversion:
      float k00 = 1/repro_matrix(2,3);
      float baseline = 1/repro_matrix(3,2);
      float mDisparityFactor = 1/k00/baseline;
      float thresh = 16.0/mDisparityFactor/depth_threshold_;
      miu_.removeSmall(d_, thresh, size_threshold_);
    }

    //std::copy(msg->images[1].data.data()             , msg->images[1].data.data() + (msg->images[1].size) ,
    //          disparity_orig_temp.data);

    // disparity_orig_temp.data = msg->images[1].data.data();   // ... is a simple assignment possible?
//    cv::Mat_<float> disparity_orig(h, w);
//    disparity_orig = disparity_orig_temp;
//
//    disparity_buf_.resize(h * w);

    cv::Mat_<float> disparity(h, w);
//    d_.convertTo(disparity,CV_32F,1./16.);
    disparity = d_;
    disparity = disparity/16.0;

    // Allocate buffer for reprojection output
//    points_buf_.resize(h * w);
    cv::Mat_<cv::Vec3f> points(h, w);

    // Do the reprojection in open space
    static const bool handle_missing_values = true;
    cv::reprojectImageTo3D(disparity, points, repro_matrix, handle_missing_values);
    std::vector<cv::Mat> xyz;
    cv::split(points,xyz);

  double min,max;
//  cv::minMaxLoc(xyz[2],&min,&max);
//  cout<<"min value: "<<min<<" max "<<max<<endl;
//  cv::imshow("z",xyz[2]); cv::Mat zz; xyz[2].copyTo(zz);
//  showNans(zz); cv::imshow("nan",zz);
//  cv::imshow("small",xyz[2]>=(max-10.));
  cv::Mat Z;
  xyz[2].copyTo(Z,xyz[2]<(max-10.));
  cv::imshow("z",Z);
  cv::waitKey(0);
  cv::minMaxLoc(Z,&min,&max);
  cout<<"min value: "<<min<<" max "<<max<<endl;
//
////  pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud(
////      new pcl::PointCloud<pcl::PointXYZRGB>);
////  ms_utils_->unpack_multisense(msg,Q_,cloud);  
//
//
  Z = Z*1000.; // because code expects it in mm
  pRtmf_->compute(Z);
////  this->update();
////  cout<<pRtmf_->cRmf()<<endl;
////
 visualizeNormals(); 
 cv::waitKey(10);
}

void RealtimeMF_lcm::visualizeNormals()
{
  static int frameNr = 0;
  cout<<"visualize Normals"<<endl;
  cv::Mat Iseg = pRtmf_->overlaySeg(this->rgb_);
  cv::imshow("seg",Iseg);
  cv::Mat n = pRtmf_->normalsImg();
  cv::Mat d = pRtmf_->smoothDepthImg();
  cv::imshow("dS",d);
  cv::imshow("nS",n);
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

