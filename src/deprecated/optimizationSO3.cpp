#include <optimizationSO3.hpp>

void OptSO3::init()
{
  //checkCudaErrors(cudaMalloc((void **)&d_q_, w*h * 3 * sizeof(float)));
  cout<<"allocating CUDA mem for OptSO3 "<<d_q_<<" ("<<w_<<"x"<<h_<<")"<<endl;
  checkCudaErrors(cudaMalloc((void **)&d_z, w_*h_ * sizeof(uint16_t)));
  Matrix<uint16_t,Dynamic,Dynamic> z = 
    Matrix<uint16_t,Dynamic,Dynamic>::Zero(h_,w_);

  checkCudaErrors(cudaMemcpy(d_z, z.data(), w_ * h_ * sizeof(uint16_t),
        cudaMemcpyHostToDevice));

  checkCudaErrors(cudaMalloc((void **)&d_errs_, w_*h_ * sizeof(float)));
  checkCudaErrors(cudaMalloc((void **)&d_cost, 6*sizeof(float)));
  checkCudaErrors(cudaMalloc((void **)&d_J, 3*3*sizeof(float)));
  checkCudaErrors(cudaMalloc((void **)&d_mu_, 6*3*sizeof(float)));
  checkCudaErrors(cudaMalloc((void **)&d_N_, sizeof(int)));
  h_z = (uint16_t *)malloc(w_ *h_ * sizeof(uint16_t));
  h_errs_ = (float *)malloc(w_ *h_ * sizeof(float));

  loadRGBvaluesForMFaxes();
};

OptSO3::~OptSO3()
{
  //    checkCudaErrors(cudaFree(d_q_));
  checkCudaErrors(cudaFree(d_z));
  checkCudaErrors(cudaFree(d_errs_));
  checkCudaErrors(cudaFree(d_cost));
  checkCudaErrors(cudaFree(d_J));
  checkCudaErrors(cudaFree(d_mu_));
  checkCudaErrors(cudaFree(d_N_));
  free(h_z);
  free(h_errs_);
};

double OptSO3::D_KL_axisUnif()
{
  checkCudaErrors(cudaDeviceSynchronize());
  checkCudaErrors(cudaMemcpy(h_z, d_z, w_*h_ *sizeof(uint16_t), 
        cudaMemcpyDeviceToHost));
  MatrixXd u(6,1);
  u << 1./6., 1./6., 1./6., 1./6., 1./6., 1./6.;
  MatrixXd w(6,1);
  w << 1.,1.,1.,1.,1.,1.;
  double N=6.0;
  for(int i=0; i<w_*h_; ++i)
    if (h_z[i] < 6)
    {
      w(h_z[i],0) ++;
      N ++;
    }
  w /= N;
  //    cout<<"w="<<w<<endl;
  //    cout<<"u="<<u<<endl;
  return (u.array()*(u.array()/w.array()).log()).sum();
};

float* OptSO3::getErrs(int w, int h)
{
  checkCudaErrors(cudaMemcpy(d_errs_, h_errs_, w * h * sizeof(float),
        cudaMemcpyHostToDevice));
  return h_errs_;
}

void OptSO3::updateGandH(Matrix3f& G, Matrix3f& G_prev, Matrix3f& H, 
    const Matrix3f& R, const Matrix3f& J, const Matrix3f& M_t_min,
    bool resetH)
{
  G_prev = G;
  G = J - R * J.transpose() * R;
  G = R*enforceSkewSymmetry(R.transpose()*G);

  if(resetH)
  {
    H= -G;
  }else{
    Matrix3f tauH = H * M_t_min; //- R_prev * (RR.transpose() * N_t_min);
    float gamma = ((G-G_prev)*G).trace()/(G_prev*G_prev).trace();
    H = -G + gamma * tauH;
    H = R*enforceSkewSymmetry(R.transpose()*H);
  }
}

#define DEBUG

float OptSO3::conjugateGradientPreparation_impl(Matrix3f& R, uint32_t& N)
{
  N = 0;
  return computeAssignment(R,N)/float(N);
};

float OptSO3::conjugateGradientCUDA_impl(Matrix3f& R, float res0, uint32_t N,
    uint32_t maxIter)
{
  Timer t0;
  Matrix3f G_prev, G, H, M_t_min, J;
//  Matrix3f R = R0;
  vector<float> res(1,res0);

#ifndef NDEBUG
  cout<<"N="<<N<<endl;
  cout<<"R0="<<endl<<R<<endl;
  cout<<"residual 0 = "<<res[0]<<endl;
#endif

  //float ts[10] = {0.0,0.1,0.2,0.3,0.4,0.5,0.7,1.0,1.5,2.0};
  //float ts[10] = {0.0,0.01,0.02,0.03,0.04,0.05,0.07,.1,.2,.3};
  Timer t1;
  for(uint32_t i =0; i<maxIter; ++i)
  {
    computeJacobian(J,R,N);
#ifndef NDEBUG
  cout<<"J="<<endl<<J<<endl;
#endif
    t0.toctic("--- jacobian ");
    updateGandH(G,G_prev,H,R,J,M_t_min,i%3 == 0); // want first iteration to reset H
#ifndef NDEBUG
  cout<<(i%3 == 2)<<endl;
  cout<<"R="<<endl<<R<<endl;
  cout<<"G="<<endl<<G<<endl;
  cout<<"H="<<endl<<H<<endl;
#endif
    t0.toctic("--- update G and H ");
    float f_t_min = linesearch(R,M_t_min,H,N,t_max_,dt_);
    t0.toctic("--- linesearch ");
    if(f_t_min == 999999.0f) break;

    res.push_back(f_t_min);
    float dresidual = res[res.size()-2] - res[res.size()-1];
    if( abs(dresidual) < 1e-7 )
    {
      cout<<"converged after "<<res.size()<<" delta residual "
        << dresidual <<" residual="<<res[res.size()-1]<<endl;
      break;
    }else{
      cout<<"delta residual " << dresidual
        <<" residual="<<res[res.size()-1]<<endl;
    }
  }
  dtCG_ = t1.toc();
//  R0 = R; // update R
  return  res.back();
}

float OptSO3::linesearch(Matrix3f& R, Matrix3f& M_t_min, const Matrix3f& H, 
    float N, float t_max, float dt)
{
  Matrix3f A = R.transpose() * H;

  EigenSolver<MatrixXf> eig(A);
  MatrixXcf U = eig.eigenvectors();
  MatrixXcf invU = U.inverse();
  VectorXcf d = eig.eigenvalues();
#ifndef NDEBUG
  cout<<"A"<<endl<<A<<endl;
  cout<<"U"<<endl<<U<<endl;
  cout<<"d"<<endl<<d<<endl;
#endif

  Matrix3f R_t_min=R;
  float f_t_min = 999999.0f;
  float t_min = 0.0f;
  //for(int i_t =0; i_t<10; i_t++)
  for(float t =0.0f; t<t_max; t+=dt)
  {
    //float t= ts[i_t];
    VectorXcf expD = ((d*t).array().exp());
    MatrixXf MN = (U*expD.asDiagonal()*invU).real();
    Matrix3f R_t = R*MN.topLeftCorner(3,3);

    float detR = R_t.determinant();
    float maxDeviationFromI = ((R_t*R_t.transpose() 
          - Matrix3f::Identity()).cwiseAbs()).maxCoeff();
    if ((R_t(0,0)==R_t(0,0)) 
        && (abs(detR-1.0f)< 1e-2) 
        && (maxDeviationFromI <1e-1))
    {
      float f_t = evalCostFunction(R_t)/float(N);
#ifndef NDEBUG
      cout<< " f_t = "<<f_t<<endl;
#endif
      if (f_t_min > f_t && f_t != 0.0f)
      {
        R_t_min = R_t;
        M_t_min = MN.topLeftCorner(3,3);
        f_t_min = f_t;
        t_min = t;
      }
    }else{
      cout<<"R_t is corruputed detR="<<detR
        <<"; max deviation from I="<<maxDeviationFromI 
        <<"; nans? "<<R_t(0,0)<<" f_t_min="<<f_t_min<<endl;
    }
  }
  if(f_t_min == 999999.0f) return f_t_min;
  // case where the MN is nans
  R = R_t_min;
#ifndef NDEBUG
#endif
  cout<<"R: det(R) = "<<R.determinant()<<endl<<R<<endl;
  cout<< "t_min="<<t_min<<" f_t_min="<<f_t_min<<endl;
  return f_t_min; 
}

float OptSO3::evalCostFunction(Matrix3f& R)
{
  Rot2Device(R);

  float residuals[6]; // for all 6 different axes
  robustSquaredAngleCostFctGPU(residuals, d_cost, d_q_, d_weights_, d_z, d_mu_, 
      sigma_sq_, w_, h_);

  float residual = 0.0f;
  for (uint32_t i=0; i<6; ++i)
  {
    //cout<<residuals[i]<<" ";
    residual +=  residuals[i];
  } //cout<<endl;
  return residual;
};

void OptSO3::computeJacobian(Matrix3f&J, Matrix3f& R, float N)
{  
  Rot2Device(R);
  //cout<<"computeJacobian"<<endl;
  J = Matrix3f::Zero();
  robustSquaredAngleCostFctJacobianGPU(J.data(), d_J,
      d_q_, d_weights_, d_z, d_mu_, sigma_sq_, w_,h_);
  J /= N;
};

float OptSO3::computeAssignment(Matrix3f& R, uint32_t& N)
{
  Rot2Device(R);

  float residuals[6]; // for all 6 different axes
  robustSquaredAngleCostFctAssignmentGPU(residuals, d_cost, &N, d_N_, d_q_,
      d_weights_, d_errs_, d_z, d_mu_, sigma_sq_, w_, h_);

  float residual = 0.0f;
  for (uint32_t i=0; i<6; ++i)
  {
    //cout<<residuals[i]<<" ";
    residual +=  residuals[i];
  } //cout<<endl;
  return residual;
};

void OptSO3::Rot2Device(Matrix3f& R)
{
  float mu[3*6];
  Rot2M(R,mu);
  checkCudaErrors(cudaMemcpy(d_mu_, mu, 3*6 * sizeof(float),
        cudaMemcpyHostToDevice));
  //    cout<<"Rot2Device"<<endl;
  //    cout<<R<<endl;
  //    Map<Matrix<float,3,6,RowMajor> > muMat(mu);
  //    cout<<muMat<<endl;
}

void OptSO3::Rot2M(Matrix3f& R, float *mu)
{
  for(uint32_t k=0; k<6; ++k){
    int j = k/2; // which of the rotation columns does this belong to
    float sign = (- float(k%2) +0.5f)*2.0f; // sign of the axis
    mu[k] = sign*R(0,j);
    mu[k+6] = sign*R(1,j);
    mu[k+12] = sign*R(2,j);
  }
};

// --------- deprecated --------------
void OptSO3::rectifyRotation(Matrix3f& R)
{
  float detR = R.determinant();
  if (abs(detR-1.0) <1e-6) return;
  // use projection of R onto SO3 to rectify the rotation matrix

  //cout<<"det(R)="<<R.determinant()<<endl<<R<<endl;

  Matrix3f M = R.transpose()*R;
  EigenSolver<Matrix3f> eig(M);
  Matrix3cf U = eig.eigenvectors();
  Vector3cf d = eig.eigenvalues();
  if (d(2).real() > 1e-6)
  {
    // http://lcvmwww.epfl.ch/new/publications/data/articles/63/simaxpaper.pdf
    // Eq. (3.7)
    // Moakher M (2002). "Means and averaging in the group of rotations."
    d = ((d.array().sqrt()).array().inverse());
    Matrix3cf D = d.asDiagonal();
    D(2,2) *= detR>0.0?1.0f:-1.0f;
    R = R*(U*D*U.transpose()).real();
  }else{
    //http://www.ti.inf.ethz.ch/ew/courses/GCMB07/material/lecture03/HornOrthonormal.pdf
    //Horn; Closed-FormSolutionofAbsoluteOrientation UsingOrthonormalMatrices
    d = ((d.array().sqrt()).array().inverse());
    d(2) = 0.0f;
    Matrix3cf Sp = d.asDiagonal(); 
    JacobiSVD<Matrix3f> svd(R*Sp.real());
    R = R*Sp.real() + (detR>0.0?1.0f:-1.0f)*svd.matrixU().col(2)*svd.matrixV().col(2).transpose();
  }
  //    Matrix3d M = (R.transpose()*R).cast<double>();
  //    EigenSolver<Matrix3d> eig(M);
  //    MatrixXd U = eig.eigenvectors();
  //    Matrix3d D = eig.eigenvalues();

  //cout<<"det(R)="<<R.determinant()<<endl<<R<<endl;

  //      R.col(0).normalize();
  //      R.col(2) = R.col(0).cross(R.col(1));
  //      R.col(2).normalize();
  //      R.col(1) = R.col(2).cross(R.col(0));
  //cout<<"det(R)="<<R.determinant()<<endl<<R<<endl;
} 
