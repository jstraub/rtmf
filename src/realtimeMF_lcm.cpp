/* Copyright (c) 2015, Julian Straub <jstraub@csail.mit.edu> Licensed
 * under the MIT license. See the license file LICENSE.
 */


#include <iostream>
#include <fstream>
#include <string>

#include <lcmtypes/kinect_depth_msg_t.h>
#include <lcmtypes/kinect_frame_msg_t.h>

#include <lcm/lcm-cpp.hpp>

#include "lcmtypes/kinect.hpp"

#include <rtmf/rtmf.hpp>

#include <zlib.h>
#include <glib.h>
#include <rtmf/jpeg-utils-ijg.h>

class RealtimeMF_lcm
{
  public:
    RealtimeMF_lcm(shared_ptr<lcm::LCM>& lcm, 
        shared_ptr<RealtimeMF>& pRtmf, string outPath="");
    virtual ~RealtimeMF_lcm();

    virtual void rgbd_cb(const lcm::ReceiveBuffer* rbuf, const
        string& channel, const  kinect::frame_msg_t* msg);

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
  lcm_->subscribe( "KINECT_FRAME",&RealtimeMF_lcm::rgbd_cb,this);
};

RealtimeMF_lcm::~RealtimeMF_lcm()
{
}

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

#include <boost/program_options.hpp>
namespace po = boost::program_options;


int main(int argc, char ** argv) {

  // Declare the supported options.
  po::options_description desc("Allowed options");
  desc.add_options()
    ("help,h", "produce help message")
    ("mode,m", po::value<string>(), 
    "mode of the rtmf (direct, approx, vmf)")
//    ("in,i", po::value<string>(), "path to input file")
    ("out,o", po::value<string>(), "path to output file")
    ("display,d", "display results")
    ("B,B", po::value<int>(), "B for guided filter")
    ("T,T", po::value<int>(), "number of iterations")
    ("eps", po::value<float>(), "eps for guided filter")
    ("f_d,f", po::value<float>(), "focal length of depth camera")
    ("nCGIter", po::value<int>(), "max number of CG iterations")
    ("dt", po::value<float>(), "steplength for linesearch")
    ("tMax", po::value<float>(), "max length for linesearch")
    ;

  po::variables_map vm;
  po::store(po::parse_command_line(argc, argv, desc), vm);
  po::notify(vm);    

  if (vm.count("help")) {
    cout << desc << "\n";
    return 1;
  }

  string mode = "vmf";
  string outPath = "";
  cudaPcl::CfgSmoothNormals cfgNormals;
  cfgNormals.f_d = 540.;
  cfgNormals.eps = 0.2*0.2;
  cfgNormals.B = 9;
  cfgNormals.compress = true;
  uint32_t T = 10;
  CfgOptSO3 cfgOptSO3;
  cfgOptSO3.sigma = 5.0f*M_PI/180.0;
  if(vm.count("mode")) mode = vm["mode"].as<string>();
  if(vm.count("out")) outPath = vm["out"].as<string>();
  if(vm.count("eps")) cfgNormals.eps = vm["eps"].as<float>();
  if(vm.count("f_d")) cfgNormals.f_d = vm["f_d"].as<float>();
  if(vm.count("B")) cfgNormals.B = uint32_t( vm["B"].as<int>());
  if(vm.count("T")) T = vm["T"].as<int>();

  if(mode.compare("direct") == 0)
  {
    cfgOptSO3.nCGIter = 10; // cannot do that many iterations
    cfgOptSO3.tMax = 1.f;
    cfgOptSO3.dt = 0.1f;
  }else if (mode.compare("approx") == 0){
    cfgOptSO3.nCGIter = 25;
    cfgOptSO3.tMax = 5.f;
    cfgOptSO3.dt = 0.05f;
  }else if (mode.compare("vmf") == 0){
    cfgOptSO3.nCGIter = 25;
    cfgOptSO3.tMax = 5.f;
    cfgOptSO3.dt = 0.05f;
  }

  if(vm.count("tMax")) cfgOptSO3.tMax  = vm["tMax"].as<float>();
  if(vm.count("dt")) cfgOptSO3.dt = vm["dt"].as<float>();
  if(vm.count("nCGIter")) cfgOptSO3.nCGIter = vm["nCGIter"].as<int>();
  findCudaDevice(argc,(const char**)argv);
  shared_ptr<RealtimeMF> pRtmf(new RealtimeMF(mode,cfgOptSO3,cfgNormals));

  shared_ptr<lcm::LCM> lcm(new lcm::LCM);
  if(!lcm->good()){
    cerr <<"ERROR: lcm is not good()" <<endl;
  }
  
  RealtimeMF_lcm app(lcm,pRtmf,outPath);
  while(0 == lcm->handle());
  return 0;
}

