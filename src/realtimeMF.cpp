/* Copyright (c) 2015, Julian Straub <jstraub@csail.mit.edu> Licensed
 * under the MIT license. See the license file LICENSE.
 */
#include <iostream>
#include <fstream>
#include <string>

#include <rtmf/rtmf.hpp>
#include <rtmf/realtimeMF_openni.hpp>


#include <boost/program_options.hpp>
namespace po = boost::program_options;

int main (int argc, char** argv)
{

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
  shared_ptr<RealtimeMF> pRtmf;
  std::vector<shared_ptr<RealtimeMF> > pRtmfs;

  if (path.length() == 0)
  {
    cout<<"no input path -> trying openni"<<endl;
    RealtimeMF_openni rtmfOpenni(pRtmf);
    rtmfOpenni.run();
    cout<<cudaDeviceReset()<<endl;
    return (0);
  }else{
    cout<<"reading depth image from "<<path<<endl;
    cv::Mat depth = cv::imread(path, 
        CV_LOAD_IMAGE_ANYDEPTH);
    cout<<"type: "<<int(depth.type()) <<" != "<<int(CV_16U) <<endl;

    if(vm.count("display")) 
    {
      cv::Mat dI(depth.rows,depth.cols,CV_8UC1);
      depth.convertTo(dI,CV_8UC1,255./4000.,-19.);
      cv::imshow("d",dI);
      cv::waitKey(0);
    }

    if (mode.compare("mmfvmf") == 0) {
      uint32_t runs = 11;
      for (uint32_t t=0; t<runs; ++t) {
        pRtmfs.push_back(shared_ptr<RealtimeMF>(new
              RealtimeMF(mode,cfgOptSO3,cfgNormals)));
        for(uint32_t i=0; i<T; ++i)
          pRtmfs.back()->compute(reinterpret_cast<uint16_t*>(depth.data),
              depth.cols,depth.rows);
      }
      Eigen::VectorXi mfCounts = Eigen::VectorXi::Zero(6);
      for (uint32_t t=0; t<runs; ++t) {
        mfCounts[pRtmfs[t]->cRmfs().size()] ++; 
      }
      cout<<"image from "<<path<<endl;
      std::cout << "histogram over MF counts: " << mfCounts.transpose()
        << std::endl;
      uint32_t mlMfCount = 1;
      mfCounts.maxCoeff(&mlMfCount);
      std::cout << "most likely " << mlMfCount << " MFs" << std::endl;
      for (uint32_t t=0; t<runs; ++t)
        if (pRtmfs[t]->cRmfs().size() == mlMfCount) { 
          pRtmf = pRtmfs[t];
          break;
        }
    } else {
      pRtmf = shared_ptr<RealtimeMF>(new RealtimeMF(mode,cfgOptSO3,cfgNormals));
      for(uint32_t i=0; i<T; ++i)
        pRtmf->compute(reinterpret_cast<uint16_t*>(depth.data),
            depth.cols,depth.rows);
    }

    cv::Mat dI = pRtmf->smoothDepthImg();
    cv::Mat nI = pRtmf->normalsImg();
    cv::Mat zI = pRtmf->labelsImg();

    string pathRgb(path);
    pathRgb.replace(path.length()-5,1,"rgb");
    cout<<"reading rgb image from "<<pathRgb<<endl;
    cv::Mat gray = cv::imread(pathRgb, CV_LOAD_IMAGE_GRAYSCALE);
//    std::vector<cv::Mat> grays(3);
//    grays.at(0) = gray;
//    grays.at(1) = gray;
//    grays.at(2) = gray;
//    cv::Mat rgb;
//    cv::merge(grays, rgb);
//
//    cv::Mat Iout;
//    cv::addWeighted(rgb , 0.7, zI, 0.3, 0.0, Iout);
//    projectDirections(Iout,pRtmf->mfAxes(),cfgNormals.f_d,pRtmf->mfAxCols_);
    cv::Mat Iout = pRtmf->overlaySeg(gray);

    if(vm.count("out"))
    {
      cv::imwrite(vm["out"].as<string>()+"_rgbLabels.png",Iout);
      ofstream out((vm["out"].as<string>()+"_cRmf.csv").data(),
          ofstream::out);
      std::vector<Eigen::Matrix3f> Rs = pRtmf->cRmfs();
      for(uint32_t i=0; i<3;++i) {
        for (uint32_t k=0; k<Rs.size(); ++k) {
          for(uint32_t j=0; j<3;++j) out << Rs[k](i,j)<<" ";
//          out << Rs[k](i,2);
        }
        out << std::endl;
      }
      out.close();
      ofstream outf((vm["out"].as<string>()+"_f.csv").data(),
          ofstream::out);
      outf << pRtmf->cost() << std::endl;
      outf.close();
      ofstream outNs((vm["out"].as<string>()+"_Ns.csv").data(),
          ofstream::out);
      for (uint32_t k=0; k<pRtmf->counts().rows()-1; ++k) 
        outNs << pRtmf->counts()(k) << " ";
      outNs << pRtmf->counts()(pRtmf->counts().rows()-1) << std::endl;
      for (uint32_t k=0; k<Rs.size()-1; ++k) 
        outNs << pRtmf->counts().middleRows(6*k,6).sum() << " ";
      outNs << pRtmf->counts().middleRows(6*(Rs.size()-1),6).sum() << std::endl;
      outNs.close();
    }

    if(vm.count("display")) 
    {
      cv::imshow("dS",dI);
      cv::imshow("normals",nI);
      cv::imshow("zI",zI);
      cv::imshow("out",Iout);
      cv::waitKey(0);
    }

    cout<<cudaDeviceReset()<<endl;
  }
  return (0);
}
