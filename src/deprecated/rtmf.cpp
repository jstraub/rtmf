#include <rtmf.hpp>

RealtimeMF::RealtimeMF(std::string mode, uint32_t w, uint32_t h) 
  :   dtNormals_(0.0), dtPrep_(0.0), dtCG_(0.0), dtTotal_(0.0),
  residual_(0.0), D_KL_(0.0), nCGIter_(10),
  tLog_("./timer.log",3,"TimerLog"), 
  w_(w), h_(h),
  normalExtractor_(570.f,w_,h_),
  fout_("./stats.log",ofstream::out),
  update(false), updateRGB_(false), 
  mode_(mode), optSO3_(NULL)
{
  float theta = 0.1;
  kRw_<< 1,0,0,
    0,cos(theta),-sin(theta),
    0,sin(theta),cos(theta);

  kRw_ << 0.99942, -0.0300857, -0.0159247,
       0.0211853,   0.915914 , -0.400816,
       0.0266444,   0.400246 ,   0.91602;
  init(w_,h_);
}

RealtimeMF::~RealtimeMF()
{
  delete optSO3_;
  fout_.close();
}


void RealtimeMF::init(int w, int h)
{
  if(optSO3_ != NULL) return;
  cout<<"inititalizing optSO3"<<endl;
  // TODO: use weights
  // TODO: make sure N is float all over the place
  //optSO3_ = new OptSO3(25.0f*M_PI/180.0f,d_n,w,h,NULL);//,d_weights);
  if(mode_.compare("direct") == 0)
  {
#ifndef WEIGHTED
    optSO3_ = new OptSO3(25.0f*M_PI/180.0f,normalExtractor_.d_normals(),w,h);//,d_weights);
#else
    optSO3_ = new OptSO3(25.0f*M_PI/180.0f,normalExtractor_.d_normals(),w,h,
        normalExtractor_.d_weights());
#endif
    nCGIter_ = 10; // cannot do that many iterations
  }else if (mode_.compare("approx") == 0){
#ifndef WEIGHTED
    optSO3_ = new OptSO3Approx(25.0f*M_PI/180.0f,normalExtractor_.d_normals(),w,h);//,d_weights);
#else
    optSO3_ = new OptSO3Approx(25.0f*M_PI/180.0f,normalExtractor_.d_normals(),w,h,
        normalExtractor_.d_weights());
#endif
    nCGIter_ = 25;
  }
}

Matrix3f RealtimeMF::depth_cb(const uint16_t *data, int w, int h) 
{
  assert(w_==w); assert(h_==h);
  tLog_.tic(-1); // reset all timers
  normalExtractor_.compute(data,w,h);
  dtNormals_ = tLog_.toc(0); 
  n_cp_ = normalExtractor_.normals();

#ifndef NDEBUG
  cout<<"OptSO3: sigma="<<optSO3_->sigma_sq_<<endl;
#endif
  Matrix3f kRwBefore = kRw_;
//  cout<<"kRw_ before = \n"<<kRw_<<endl;
  tLog_.tic(1);
//  init(w,h); // do init for optSO3 
  double residual = optSO3_->conjugateGradientCUDA(kRw_,nCGIter_);
  tLog_.toc(1);
  double D_KL = optSO3_->D_KL_axisUnif();
  dtPrep_ = optSO3_->dtPrep();
  dtCG_ = optSO3_->dtCG();

  boost::mutex::scoped_lock updateLock(updateModelMutex);
  D_KL_= D_KL;
  residual_ = residual;
  nDisp_ = pcl::PointCloud<pcl::PointXYZRGB>(*n_cp_); // copy point cloud

  checkCudaErrors(cudaDeviceSynchronize());
  //memcpy(h_dbg,data,w*h*sizeof(float));
//#ifdef SHOW_WEIGHTS
//  checkCudaErrors(cudaMemcpy(h_dbg, d_weights, w*h *sizeof(float), 
//        cudaMemcpyDeviceToHost));
//#endif
//#ifdef SHOW_LOW_ERR
//  checkCudaErrors(cudaMemcpy(h_dbg, optSO3_->d_errs_, w*h *sizeof(float), 
//        cudaMemcpyDeviceToHost));
//#endif
//  checkCudaErrors(cudaDeviceSynchronize());
//  checkCudaErrors(cudaMemcpy(h_n, d_n, w*h* X_STEP *sizeof(float), 
//        cudaMemcpyDeviceToHost));
//  checkCudaErrors(cudaDeviceSynchronize());
//  checkCudaErrors(cudaMemcpy(h_xyz, d_xyz, w*h*4 *sizeof(float), 
//        cudaMemcpyDeviceToHost));
//  checkCudaErrors(cudaDeviceSynchronize());
  // update viewer
  update = true;
  updateLock.unlock();

  tLog_.toc(2); // total time
  tLog_.logCycle();
  cout<<"delta rotation kRw_ = \n"<<kRwBefore*kRw_.transpose()<<endl;
  cout<<"---------------------------------------------------------------------------"<<endl;
  tLog_.printStats();
  cout<<" residual="<<residual_<<"\t D_KL="<<D_KL_<<endl;
  cout<<"---------------------------------------------------------------------------"<<endl;
  fout_<<D_KL_<<" "<<residual_<<endl; fout_.flush();

  return kRw_;
}

void RealtimeMF::visualizePc()
{
  // Block signals in this thread
  sigset_t signal_set;
  sigaddset(&signal_set, SIGINT);
  sigaddset(&signal_set, SIGTERM);
  sigaddset(&signal_set, SIGHUP);
  sigaddset(&signal_set, SIGPIPE);
  pthread_sigmask(SIG_BLOCK, &signal_set, NULL);

  bool showNormals =true;
  float scale = 2.0f;
  // prepare visualizer named "viewer"
  boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer (
      new pcl::visualization::PCLVisualizer ("3D Viewer"));

  //      viewer->setPointCloudRenderingProperties (
  //          pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "sample cloud");
  viewer->initCameraParameters ();

  cv::namedWindow("normals");
  //cv::namedWindow("dbg");
  cv::namedWindow("dbgNan");
//  cv::namedWindow("rgb");

  int v1(0);
  viewer->createViewPort (0.0, 0.0, 0.5, 1.0, v1);
  viewer->setBackgroundColor (0, 0, 0, v1);
  viewer->addText ("normals", 10, 10, "v1 text", v1);
  viewer->addCoordinateSystem (1.0,v1);

  int v2(0);
  viewer->createViewPort (0.5, 0.0, 1.0, 1.0, v2);
  viewer->setBackgroundColor (0.1, 0.1, 0.1, v2);
  viewer->addText ("pointcloud", 10, 10, "v2 text", v2);
  viewer->addCoordinateSystem (1.0,v2);

  pcl::PointCloud<pcl::PointXYZRGB>::Ptr n;
  pcl::PointCloud<pcl::PointXYZ>::Ptr pc;
  cv::Mat Icomb;

  Timer t;
  while (!viewer->wasStopped ())
  {
    cout<<"viewer"<<endl;
    viewer->spinOnce (10);
    cv::waitKey(10);
    // break, if the last update was less than 2s ago
//    if (t.dtFromInit() > 20000.0)
//    {
//      cout<<" ending visualization - waited too long"<<endl;
//      break;
//    }
//    cout<<" after break"<<endl;

    // Get lock on the boolean update and check if cloud was updated
    boost::mutex::scoped_lock updateLock(updateModelMutex);
//    if (updateRGB_)
//    {
//      cout<<"show rgb"<<endl;
//      imshow("rgb",rgb_);
//      updateRGB_ = false;
//      t=Timer();
//    } 
    if (update)
    {
      cout<<"show pc"<<endl;
      stringstream ss;
      ss <<"residual="<<residual_;
      if(!viewer->updateText(ss.str(),10,20,"residual"))
        viewer->addText(ss.str(),10,20, "residual", v1);
      ss.str(""); ss << "D_KL="<<D_KL_;
      if(!viewer->updateText(ss.str(),10,30,"D_KL"))
        viewer->addText(ss.str(),10,30, "D_KL", v1);

      Matrix3f wRk = kRw_.transpose();
      Matrix4f wTk;
      wTk<< wRk, MatrixXf::Zero(3,1),MatrixXf::Zero(1,3),1.0;

      cv::Mat nI(nDisp_.height,nDisp_.width,CV_32FC3); 
      for(uint32_t i=0; i<nDisp_.width; ++i)
        for(uint32_t j=0; j<nDisp_.height; ++j)
        {
          // nI is BGR but I want R=x G=y and B=z
          nI.at<cv::Vec3f>(j,i)[0] = (1.0f+nDisp_.points[i+j*nDisp_.width].z)*0.5f; // to match pc
          nI.at<cv::Vec3f>(j,i)[1] = (1.0f+nDisp_.points[i+j*nDisp_.width].y)*0.5f; 
          nI.at<cv::Vec3f>(j,i)[2] = (1.0f+nDisp_.points[i+j*nDisp_.width].x)*0.5f; 
        }
      cv::imshow("normals",nI); 

      union{
        uint32_t asInt;
        float asFloat;
      } rgb;
      cv::Mat zI(nDisp_.height,nDisp_.width,CV_8UC3); 
      for(uint32_t i=0; i<nDisp_.width; ++i)
        for(uint32_t j=0; j<nDisp_.height; ++j)
        {
          // nI is BGR but I want R=x G=y and B=z
          rgb.asFloat = nDisp_.points[i+j*nDisp_.width].rgb;
          zI.at<cv::Vec3b>(j,i)[0] = uint8_t(( rgb.asInt >> 16) & 255); // to match pc
          zI.at<cv::Vec3b>(j,i)[1] = uint8_t((rgb.asInt  >> 8) & 255); 
          zI.at<cv::Vec3b>(j,i)[2] = uint8_t((rgb.asInt  & 255));
        }
      cv::addWeighted(rgb_ , 0.7, zI, 0.3, 0.0, Icomb);                      
      cv::imshow("dbg",Icomb);     

//      cv::Mat Ierr(nDisp_.height,nDisp_.width,CV_32FC1); 
//      for(uint32_t i=0; i<nDisp_.width; ++i)
//        for(uint32_t j=0; j<nDisp_.height; ++j)
//        {
//          // nI is BGR but I want R=x G=y and B=z
//          nI.at<float>(j,i) = h_n[(i+j*nDisp_.width)*X_STEP+6];
//        }
//      normalizeImg(Ierr);

//#ifdef SHOW_WEIGHTS
//              cv::Mat Idbg(nDisp_.height,nDisp_.width,CV_32FC1,h_dbg);
//              normalizeImg(Idbg);
//              cv::imshow("dbg",Idbg);
//#endif
      //        cv::Mat IdbgNan = Idbg.clone();
      //        showNans(IdbgNan);
      //        //showZeros(IdbgNan);
      //        cv::imshow("dbgNan",IdbgNan);
      
#ifdef SHOW_LOW_ERR
      cv::Mat Idbg(nDisp_.height,nDisp_.width,CV_32FC1,h_dbg);
      cv::Mat IgoodRGB = cv::Mat::zeros(nDisp_.height,nDisp_.width,CV_8UC3); 
      rgb_.copyTo(IgoodRGB,Idbg < 20.0*M_PI/180.0);
      normalizeImg(Idbg);
      cv::imshow("dbgNan",IgoodRGB); // Idbg
#endif

      if(showNormals){
        n = pcl::PointCloud<pcl::PointXYZRGB>::Ptr(
            new pcl::PointCloud<pcl::PointXYZRGB>);
        for(uint32_t i=0; i<nDisp_.points.size(); i+= 5)
          if (*((&nDisp_.points[i].rgb)+1) < 6)
          {
            n->push_back(nDisp_.points[i]);
            n->back().x *= scale;
            n->back().y *= scale;
            n->back().z *= scale;
          }
        //pcl::transformPointCloud(*n, *n, wTk);
      }

      pc = pcl::PointCloud<pcl::PointXYZ>::Ptr(
          new pcl::PointCloud<pcl::PointXYZ>);
      for(uint32_t i=0; i<pc_.width; i+= 5)
        for(uint32_t j=0; j<pc_.height; j+=5)
          pc->points.push_back(pc_cp_->points[i+j*pc_.width]);
      pcl::transformPointCloud(*pc, *pc , wTk);

      if(showNormals)
        if(!viewer->updatePointCloud(n, "normals"))
          viewer->addPointCloud(n, "normals",v1);

      if(!viewer->updatePointCloud(pc, "pc"))
        viewer->addPointCloud(pc, "pc",v2);

//      viewer->saveScreenshot("./test.png");
      update = false;
      t=Timer();
    }
    updateLock.unlock();
  }
}


void RealtimeMF::run ()
{
  boost::thread visualizationThread(&RealtimeMF::visualizePc,this); 

  this->run_impl();
  while (42) boost::this_thread::sleep (boost::posix_time::seconds (1));
  this->run_cleanup_impl();
  visualizationThread.join();
}

