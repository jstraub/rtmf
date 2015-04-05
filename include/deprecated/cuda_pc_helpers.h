
#ifndef CUDA_PC_HELPERS_H
#define CUDA_PC_HELPERS_H

// step size of the normals 
// for PointXYZI
#define X_STEP 8
#define X_OFFSET 0
// for PointXYZ
//#define X_STEP 4
//#define X_OFFSET 0
//
#define BOARDER_SIZE 10


// forward declaration
extern "C" void depth2xyzGPU(unsigned short* d, float* x, float* y, float* z, 
    float invF, int w, int h,float *xyz);

extern "C" void depth2floatGPU(unsigned short* d, float* d_float, 
    int w, int h);

extern "C" void depthFilterGPU(float* d, int w, int h);

extern "C" void depth2xyzFloatGPU(float* d, float* x, float* y, float* z,
    float invF, int w, int h, float *xyz);

extern "C" void derivatives2normalsGPU(float* d_x, float* d_y, float* d_z, 
    float* d_xu, float* d_yu, float* d_zu, 
    float* d_xv, float* d_yv, float* d_zv, 
    float* d_n, int w, int h);

extern "C" void derivatives2normalsCleanerGPU(float* d_x, float* d_y, float* d_z, 
    float* d_xu, float* d_yu, float* d_zu, 
    float* d_xv, float* d_yv, float* d_zv, 
    float* d_n, int w, int h);


extern "C" void weightsFromCovGPU(float* z, float* weights,
    float theta, float invF, int w, int h);

extern "C" void weightsFromAreaGPU(float* z, float* weights,
    int w, int h);

//extern void bilateralFilterGPU(float *in, float* out, int w, int h, 
//  uint32_t radius, float sigma_spatial, float variance_I)

#endif
