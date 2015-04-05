#include <optimizationSO3_approx.hpp>

float OptSO3Approx::conjugateGradientPreparation_impl(Matrix3f& R, uint32_t& N)
{
  Timer t0;
  N =0;
  float res0 = computeAssignment(R,N)/float(N);
  if(this->t_ == 0){
   // init karcher means with columns of the rotation matrix (takes longer!)
  qKarch_ << R.col(0),-R.col(0),R.col(1),-R.col(1),R.col(2),-R.col(2);
//    // init karcher means with rotated version of previous karcher means
//    qKarch_ =  (this->Rprev_*R.transpose())*qKarch_; 
  }
  qKarch_ = karcherMeans(qKarch_, 5.e-5, 10);
  t0.toctic("----- karcher mean");
  // compute a rotation matrix from the karcher means (not necessary)
  Matrix3f Rkarch;
  Rkarch.col(0) =  qKarch_.col(0);
  Rkarch.col(1) =  qKarch_.col(2) - qKarch_.col(2).dot(qKarch_.col(0))
    *qKarch_.col(0);
  Rkarch.col(1) /= Rkarch.col(1).norm();
  Rkarch.col(2) = Rkarch.col(0).cross(Rkarch.col(1));
#ifndef NDEBUG
  cout<<"R: "<<endl<<R<<endl;
  cout<<"Rkarch: "<<endl<<Rkarch<<endl<<"det(Rkarch)="<<Rkarch.determinant()<<endl;
#endif
  cout<<"R: "<<endl<<R<<endl;
  cout<<"Rkarch: "<<endl<<Rkarch<<endl<<"det(Rkarch)="<<Rkarch.determinant()<<endl;
  t0.tic();
  computeSuffcientStatistics();
  t0.toctic("----- sufficient statistics");
  return res0; // cost fct value
}

void OptSO3Approx::conjugateGradientPostparation_impl(Matrix3f& R)
{
//  if(this->t_ > 0)
//  {
//    // init karcher means with rotated version of previous karcher means
//    qKarch_ =  (this->Rprev_*R.transpose())*qKarch_; 
//    // rotations are cameraRworld
//  }
};

/* evaluate cost function for a given assignment of npormals to axes */
float OptSO3Approx::evalCostFunction(Matrix3f& R)
{
  float c = 0.0f;
  for (uint32_t j=0; j<6; ++j)
  { 
    Matrix<float,2,1> mu;
    if(j%2 ==0){
      mu = S2_.Log_p_2D(qKarch_.col(j), R.col(j/2));
    }else{
      mu = S2_.Log_p_2D(qKarch_.col(j), -R.col(j/2));
    }
    c += 0.5f * (invSigma_*Ss_[j]).trace() 
      - mu.transpose()*(invSigma_*xSums_.col(j))
      + 0.5f*Ns_[j]*mu.transpose()*invSigma_*mu;
  }
  return c;
}
/* compute Jacobian */
void OptSO3Approx::computeJacobian(Matrix3f&J, Matrix3f& R, float N)
{
  J = Matrix3f::Zero();
#ifndef NDEBUG
  cout<<"qKarch"<<endl<<qKarch_<<endl;
  cout<<"xSums_"<<endl<<xSums_<<endl;
  cout<<"Ns_"<<endl<<Ns_<<endl;
#endif
  for (uint32_t j=0; j<6; ++j)
  {
    if(j%2 ==0){
      Matrix<float,2,1> x = invSigma_*(xSums_.col(j) -
          Ns_(j)*S2_.Log_p_2D(qKarch_.col(j),R.col(j/2)));
      J.col(j/2) -= S2_.rotate_north2p(qKarch_.col(j),x); 
    }else{
      Matrix<float,2,1> x = invSigma_*(xSums_.col(j) -
          Ns_(j)*S2_.Log_p_2D(qKarch_.col(j),-R.col(j/2)));
      //cout<<x<<endl;
      J.col(j/2) += S2_.rotate_north2p(qKarch_.col(j),x); 
    }
  }
  J /= N;
}
/* recompute assignment based on rotation R and return residual as well */
//float OptSO3Approx::computeAssignment(Matrix3f& R, int& N)
//{
//  return 0.0f;
//}

Matrix<float,3,6> OptSO3Approx::meanInTpS2_GPU(Matrix<float,3,6>& p)
{
  Matrix<float,4,6> mu_karch = Matrix<float,4,6>::Zero();
  float *h_mu_karch = mu_karch.data();
  float *h_p = p.data();
  meanInTpS2GPU(h_p, d_p_, h_mu_karch, d_mu_karch_, d_q_, d_z, d_weights_, w_, h_);
  Matrix<float,3,6> mu = mu_karch.topRows(3);
  for(uint32_t i=0; i<6; ++i)
    if(mu_karch(3,i) >0)
      mu.col(i) /= mu_karch(3,i);
  return mu;
}

Matrix<float,3,6> OptSO3Approx::karcherMeans(const Matrix<float,3,6>& p0, 
    float thresh, uint32_t maxIter)
{
  Matrix<float,3,6> p = p0;
  Matrix<float,6,1> residuals;
  for(uint32_t i=0; i< maxIter; ++i)
  {
    Timer t0;
    Matrix<float,3,6> mu_karch = meanInTpS2_GPU(p);
    t0.toctic("meanInTpS2_GPU");
#ifndef NDEBUG
    cout<<"mu_karch"<<endl<<mu_karch<<endl;
#endif
    residuals.fill(0.0f);
    for (uint32_t j=0; j<6; ++j)
    {
      p.col(j) = S2_.Exp_p(p.col(j), mu_karch.col(j));
      residuals(j) = mu_karch.col(j).norm();
    }
    cout<<"p"<<endl<<p<<endl;
    cout<<"karcherMeans "<<i<<" residuals="<<residuals.transpose()<<endl;
    if( (residuals.array() < thresh).all() )
    {
#ifndef NDEBUG
      cout<<"converged after "<<i<<" residuals="<<residuals.transpose()<<endl;
#endif
      break;
    }
  }
  return p;
}

void OptSO3Approx::computeSuffcientStatistics()
{
  // compute rotations to north pole
  Matrix<float,2*6,3,RowMajor> Rnorths(2*6,3);
  for (uint32_t j=0; j<6; ++j)
  {
    Rnorths.middleRows<2>(j*2) = S2_.north_R_TpS2(qKarch_.col(j)).topRows<2>();
    //cout<<qKarch_.col(j).transpose()<<endl;
    //cout<<Rnorths.middleRows<2>(j*2)<<endl;
    //cout<<"----"<<endl;
  }
  //cout<<Rnorths<<endl;

  Matrix<float,7,6,ColMajor> SSs;
  sufficientStatisticsOnTpS2GPU(qKarch_.data(), d_mu_karch_, 
    Rnorths.data(), d_Rnorths_, d_q_, d_z , w_, h_,
    SSs.data(), d_SSs_);
  
  //cout<<SSs<<endl; 
  for (uint32_t j=0; j<6; ++j)
  {
    xSums_.col(j) = SSs.block<2,1>(0,j);
    Ss_[j](0,0) =  SSs(2,j);
    Ss_[j](0,1) =  SSs(3,j);
    Ss_[j](1,0) =  SSs(4,j);
    Ss_[j](1,1) =  SSs(5,j);
    Ns_(j) = SSs(6,j);
    //cout<<"@j="<<j<<"\t"<< Ss_[j]<<endl;
  }
  //cout<<xSums_<<endl;
  //cout<<Ns_<<endl;
  
}
