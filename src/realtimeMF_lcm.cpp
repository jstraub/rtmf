/* Copyright (c) 2015, Julian Straub <jstraub@csail.mit.edu> Licensed
 * under the MIT license. See the license file LICENSE.
 */


#include <iostream>
#include <fstream>
#include <string>

#include <lcm/lcm.h>
#include <lcmtypes/kinect_depth_msg_t.h>
#include <lcmtypes/kinect_frame_msg_t.h>

#include <rtmf/rtmf.hpp>
#include <rtmf/realtimeMF_openni.hpp>


#include <boost/program_options.hpp>
namespace po = boost::program_options;



