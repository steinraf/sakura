//
// Created by steinraf on 20/12/22.
//

#pragma once

#include "../../src_old/cudaHelpers.cuh"
#include "../../src_old/utility/vector.cuh"


__global__ void denoiser(FeatureBuffer *featureBuffer, Vector3f *output, float *weights, int width, int height);

__global__ void denoiseApplyWeights(FeatureBuffer *featureBuffer, Vec3f *output, float *weights, int width, int height);
