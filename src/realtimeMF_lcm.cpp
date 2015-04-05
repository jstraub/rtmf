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

class RealtimeMF_lcm
{
  public:
    RealtimeMF_lcm(shared_ptr<lcm::LCM>& lcm, 
        shared_ptr<RealtimeMF>& pRtmf);
    virtual ~RealtimeMF_lcm();

    virtual void rgbd_cb(const lcm::ReceiveBuffer* rbuf, const
        string& channel, const  kinect::frame_msg_t* msg);

  protected:
    virtual void visualizeNormals();

   shared_ptr<lcm::LCM> lcm_;
   shared_ptr<RealtimeMF> pRtmf_;

   cv::Mat rgb_;
   cv::Mat d_;

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
    rgb_ = cv::Mat(msg->image.height, msg->image.width, CV_8UC3,
        const_cast<uint8_t*>(&(msg->image.image_data[0])));
  }else if (msg->image.image_data_format == kinect::image_msg_t::VIDEO_RGB_JPEG){
    cout << "jpg rgb image "<<endl;
  }else{
    cout << "Format not recognized: " << msg->image.image_data_format << endl;
  }

  if (msg->depth.depth_data_format == kinect::depth_msg_t::DEPTH_MM)
  {
    cout << "MM depth image"<<endl;
    d_ = cv::Mat(msg->depth.height, msg->depth.width, CV_16SC1,
        const_cast<uint8_t*>(&(msg->depth.depth_data[0])));
    cv::imshow("depth",d_);
  }else{
    cout << "Format not recognized: " << msg->depth.depth_data_format << endl;
  }

  pRtmf_->compute(d_);
//  this->update();
//  cout<<pRtmf_->cRmf()<<endl;
//
  cout<<"visualize Normals"<<endl;
  cv::Mat Iseg = pRtmf_->overlaySeg(this->rgb_);
  cv::imshow("seg",Iseg);
  cv::waitKey(10);
}

void RealtimeMF_lcm::visualizeNormals()
{
  cout<<"visualize Normals"<<endl;
  cv::Mat Iseg = pRtmf_->overlaySeg(this->rgb_);
  cv::imshow("seg",Iseg);
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
    ("in,i", po::value<string>(), "path to input file")
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

  string mode = "approx";
  string path = "";
  cudaPcl::CfgSmoothNormals cfgNormals;
  cfgNormals.f_d = 540.;
  cfgNormals.eps = 0.2*0.2;
  cfgNormals.B = 9;
  cfgNormals.compress = true;
  uint32_t T = 10;
  CfgOptSO3 cfgOptSO3;
  cfgOptSO3.sigma = 5.0f*M_PI/180.0;
  if(vm.count("mode")) mode = vm["mode"].as<string>();
  if(vm.count("in")) path = vm["in"].as<string>();
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
  
  RealtimeMF_lcm app(lcm,pRtmf);
  while(0 == lcm->handle());
  return 0;
}

