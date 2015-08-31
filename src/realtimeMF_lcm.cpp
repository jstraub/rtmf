/* Copyright (c) 2015, Julian Straub <jstraub@csail.mit.edu> Licensed
 * under the MIT license. See the license file LICENSE.
 */

#include <iostream>
#include <fstream>
#include <string>

#include <rtmf/realtimeMF_lcm.hpp>

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
    ("channel,c", po::value<string>(), "channel: KINECT_FRAME or CAMERA (disparity stream)")
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
  string channel= "KINECT_FRAME";
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
  if(vm.count("channel")) channel = vm["channel"].as<string>();
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
  if(channel.compare("CAMERA"))
  {
  // from drc/software/config/config_components/multisense_02.cfg 
  //
  // multisense stereo camera
    cfgNormals.f_d = 591.909423828125;
  };

  if(vm.count("tMax")) cfgOptSO3.tMax  = vm["tMax"].as<float>();
  if(vm.count("dt")) cfgOptSO3.dt = vm["dt"].as<float>();
  if(vm.count("nCGIter")) cfgOptSO3.nCGIter = vm["nCGIter"].as<int>();
  findCudaDevice(argc,(const char**)argv);
  shared_ptr<RealtimeMF> pRtmf(new RealtimeMF(mode,cfgOptSO3,cfgNormals));

  shared_ptr<lcm::LCM> lcm(new lcm::LCM);
  if(!lcm->good()){
    cerr <<"ERROR: lcm is not good()" <<endl;
  }
  
  RealtimeMF_lcm app(lcm,pRtmf,channel,outPath);
  while(0 == lcm->handle());
  return 0;
}

