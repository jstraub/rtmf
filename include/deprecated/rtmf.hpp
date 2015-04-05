#ifndef REALTIME_MF_HPP_INCLUDED
#define REALTIME_MF_HPP_INCLUDED
#include <root_includes.hpp>
#include <defines.h>
#include <signal.h>
#include <string>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <boost/thread.hpp>

#include <Eigen/Dense>

// CUDA runtime
#include <cuda_runtime.h>

// Utilities and system includes
#include <nvidia/helper_functions.h>
#include <nvidia/helper_cuda.h>

#include <cuda_pc_helpers.h>
#include <convolutionSeparable_common.h>
#include <convolutionSeparable_common_small.h>
#include <optimizationSO3.hpp>
#include <optimizationSO3_approx.hpp>
#include <cv_helpers.hpp>
#include <pcl_helpers.hpp>

//#include <normalExtractCUDA.hpp>
#include <cudaPcl/timer.hpp>
#include <cudaPcl/timerLog.hpp>

using namespace Eigen;

class RealtimeMF
{
  public:
    RealtimeMF(std::string mode, uint32_t w, uint32_t h);
    ~RealtimeMF();

    /* process a depth image of size w*h and return rotation estimate
     * mfRc
     */
    Matrix3f depth_cb(const uint16_t *data, int w,int h);

    void getAxisAssignments();

    void visualizePc();

    void run();

    double dtNormals_;
    double dtPrep_;
    double dtCG_;
    double dtTotal_;
    double residual_, D_KL_;
    uint32_t nCGIter_;

    TimerLog tLog_;

  protected:

    void init(int w, int h);

    uint32_t w_, h_;
    boost::mutex updateModelMutex;

    NormalExtractGpu<float> normalExtractor_;

    ofstream fout_;
    bool update, updateRGB_;
    std::string mode_;

    Matrix3f kRw_;

    OptSO3 *optSO3_;
    pcl::PointCloud<pcl::PointXYZRGB> nDisp_;
    pcl::PointCloud<pcl::PointXYZRGB>::ConstPtr nDisp_cp_;
    pcl::PointCloud<pcl::PointXYZRGB> n_;
    pcl::PointCloud<pcl::PointXYZRGB>::ConstPtr n_cp_;

    pcl::PointCloud<pcl::PointXYZ> pc_;
    pcl::PointCloud<pcl::PointXYZ>::ConstPtr pc_cp_;

    cv::Mat rgb_;

    Vector3d vicon_t_; 
    Quaterniond vicon_q_; 
    Matrix3d kRv;
    Vector3d dt0;

    virtual void run_impl() = 0;
    virtual void run_cleanup_impl() = 0;
};

#endif
