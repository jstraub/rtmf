/* Copyright (c) 2015, Julian Straub <jstraub@csail.mit.edu> Licensed
 * under the MIT license. See the license file LICENSE.
 */

#include <iostream>
#include <string>

#include <rtmf/realtimeMF_openni.hpp>

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
    ;

  po::variables_map vm;
  po::store(po::parse_command_line(argc, argv, desc), vm);
  po::notify(vm);    

  if (vm.count("help")) {
    cout << desc << "\n";
    return 1;
  }

  string mode = "approx";
  if(vm.count("mode")) mode = vm["mode"].as<string>();

  findCudaDevice(argc,(const char**)argv);

  shared_ptr<RealtimeMF> pRtmf(new RealtimeMF());
  RealtimeMF_openni v(mode,640,480,470.);
  v.init(640,480);
  v.run ();
  cout<<cudaDeviceReset()<<endl;
  return (0);
}
