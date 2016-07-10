/* Copyright (c) 2015, Julian Straub <jstraub@csail.mit.edu> Licensed
 * under the MIT license. See the license file LICENSE.
 */

#include <iostream>
#include <string>

#include <rtmf/realtimeMF_realsense.hpp>

// Utilities and system includes
//#include <helper_functions.h>
//#include <helper_cuda.h>

#include <boost/program_options.hpp>
namespace po = boost::program_options;

int main (int argc, char** argv)
{

  // Declare the supported options.
  po::options_description desc("Allowed options");
  desc.add_options()
    ("help,h", "produce help message")
    ("mode", po::value<string>(), 
    "mode of the rtmf (direct, approx, vmf)")
    ("B,B", po::value<int>(), "B for guided filter")
    ("T,T", po::value<int>(), "number of iterations")
    ("eps", po::value<float>(), "eps for guided filter")
    ("f_d,f", po::value<float>(), "focal length of depth camera")
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
  }else if(mode.compare("directGD") == 0)
  {
    cfgOptSO3.nCGIter = 10; // cannot do that many iterations
  }else if (mode.compare("approx") == 0){
    cfgOptSO3.nCGIter = 25;
    cfgOptSO3.tMax = 5.f;
    cfgOptSO3.dt = 0.05f;
  }else if (mode.compare("approxGD") == 0){
    cfgOptSO3.nCGIter = 25;
  }else if (mode.compare("vmf") == 0){
    cfgOptSO3.nCGIter = 25;
    cfgOptSO3.tMax = 5.f;
    cfgOptSO3.dt = 0.05f;
  }else if (mode.compare("vmfCF") == 0){
    cfgOptSO3.nCGIter = 1;
  }
  
  std::cout << "mode: " << mode << std::endl;

  if(vm.count("tMax")) cfgOptSO3.tMax  = vm["tMax"].as<float>();
  if(vm.count("dt")) cfgOptSO3.dt = vm["dt"].as<float>();
  if(vm.count("nCGIter")) cfgOptSO3.nCGIter = vm["nCGIter"].as<int>();

  findCudaDevice(argc,(const char**)argv);

  shared_ptr<RealtimeMF> pRtmf(new RealtimeMF(mode,cfgOptSO3,cfgNormals));
  RealtimeMF_realsense v(pRtmf,640,480,30);
  v.run ();
  cout<<cudaDeviceReset()<<endl;
  return (0);
}
