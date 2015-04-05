#include <iostream>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/visualization/cloud_viewer.h>
#include <pcl/features/integral_image_normal.h>

#include <timer.hpp>

int main (int argc, char** argv)
{
  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud (new pcl::PointCloud<pcl::PointXYZ>);

  if (pcl::io::loadPCDFile<pcl::PointXYZ> ("test_pcd.pcd", *cloud) == -1) //* load the file
  {
    PCL_ERROR ("Couldn't read file test_pcd.pcd \n");
    return (-1);
  }

  int w = cloud->width;
  int h = cloud->height;
  cout<<cloud->width << "x"<<cloud->height<<endl;

  cloud->points[10+w*10] = pcl::PointXYZ(1.0,2.0,3.0);
  cout<<cloud->points.data()<<endl;
  float* data = cloud->points.data()->data;
  cout<<data[(10+w*10)*3+0]<<" "<<data[(10+w*10)*3+1]<<" "<<data[(10+w*10)*3+2]<<endl;
  cout<<cloud->points[10+w*10]<<endl;

  // estimate normals
  Timer t0;
  pcl::PointCloud<pcl::Normal>::Ptr normals (new pcl::PointCloud<pcl::Normal>);
  pcl::IntegralImageNormalEstimation<pcl::PointXYZ, pcl::Normal> ne;
  ne.setNormalEstimationMethod (ne.AVERAGE_3D_GRADIENT); // 31ms
  //ne.setNormalEstimationMethod (ne.AVERAGE_DEPTH_CHANGE); // 23ms
  //ne.setNormalEstimationMethod (ne.COVARIANCE_MATRIX); // 47ms
  ne.setMaxDepthChangeFactor(0.02f);
  ne.setNormalSmoothingSize(10.0f);
  ne.setInputCloud(cloud);
  ne.compute(*normals);
  t0.toc();
  cout<<t0<<endl;

  // visualize normals
  pcl::visualization::PCLVisualizer viewer("PCL Viewer");
  viewer.setBackgroundColor (0.3, 0.3, 0.3);
  pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> single_color (cloud, 200, 10, 10);
  viewer.addPointCloud(cloud,single_color,"cloud");
  viewer.addPointCloudNormals<pcl::PointXYZ,pcl::Normal>(cloud, normals,10,0.05,"normals");
 
  while (!viewer.wasStopped ())
  {
    viewer.spinOnce ();
  }

  return 0;
}
