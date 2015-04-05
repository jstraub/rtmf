

#include <iostream>
#include <string>
#include <realtimeMF.hpp>
#include <timer.hpp>

// Utilities and system includes
#include <helper_functions.h>
#include <helper_cuda.h>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

class RealtimeMF_test: public RealtimeMF
{
  string pathToTestImg_;
  public:
    RealtimeMF_test(string mode, string pathToTestImg):
      RealtimeMF(mode), pathToTestImg_(pathToTestImg)
    {};
    ~RealtimeMF_test()
    {};

  protected:
    Matrix3f R_;

    void run_impl ()
    {
      // just load the test image and run RealtimeMF on it
      cv::Mat D = cv::imread(pathToTestImg_,CV_LOAD_IMAGE_ANYDEPTH);
      cout<<"processing: "<<pathToTestImg_<<endl;
      cout<<D.cols<<"x"<<D.rows<<endl;
      R_ = depth_cb((uint16_t*)D.data,D.cols,D.rows);
      testOptSO3Approx(D.cols,D.rows);
    }
    void run_cleanup_impl()
    {
      // nothing to do here
    }

    void testOptSO3Approx(int w, int h)
    {
      cout<<"testOptSO3Approx"<<endl;
      Matrix<float,3,6> p0;
      OptSO3Approx optSO3(0.03,optSO3_->d_q_,w,h);
      optSO3.d_z = optSO3_->d_z;
      //R_ = Matrix3f::Identity();
      p0 << R_.col(0),-R_.col(0),R_.col(1),-R_.col(1),R_.col(2),-R_.col(2);
      cout<<"running test with p0"<<endl<<p0<<endl;
      Timer t0;
      Timer t1;
//      optSO3.qKarch_ = optSO3.karcherMeans(p0,10);
//      t0.toctic("first iterations");
////      optSO3.qKarch_ = optSO3.karcherMeans(optSO3.qKarch_,10);
////      t0.toctic("after running with init from previous run");
//      optSO3.computeSuffcientStatistics();
//      t0.toctic("sufficient statistics");
      Matrix3f R2 = Matrix3f::Identity();
//      optSO3.conjugateGradientCUDA(R2);
//      t0.toctic("CG ");
//      t1.toctic("total CG ");
      optSO3.conjugateGradientCUDA(R_);
      t1.toctic("total CG ");
      cout<<p0<<endl;
      cout<<optSO3.qKarch_<<endl;
      cout<<R_<<endl;
      cout<<R2<<endl;
    }
};


int main (int argc, char** argv)
{

  findCudaDevice(argc,(const char**)argv);

  RealtimeMF_test v("approx","/data/vision/fisher/data1/mixture_of_manhattan_frames_data/2014-01-30.14:45:23_d.png");
  v.run ();
  cout<<cudaDeviceReset()<<endl;
  return (0);
}
