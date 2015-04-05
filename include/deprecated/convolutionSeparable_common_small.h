/*
 * Copyright 1993-2013 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */



#ifndef CONVOLUTIONSEPARABLE_COMMON_SMALL_H
#define CONVOLUTIONSEPARABLE_COMMON_SMALL_H



#define KERNEL_RADIUS_S 1
#define KERNEL_LENGTH_S (2 * KERNEL_RADIUS_S + 1)


////////////////////////////////////////////////////////////////////////////////
// GPU convolution
////////////////////////////////////////////////////////////////////////////////
extern "C" void setConvolutionKernel_small(float *h_Kernel);

extern "C" void convolutionRowsGPU_small(
    float *d_Dst,
    float *d_Src,
    int imageW,
    int imageH
);

extern "C" void convolutionColumnsGPU_small(
    float *d_Dst,
    float *d_Src,
    int imageW,
    int imageH
);



#endif
