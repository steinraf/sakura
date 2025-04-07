//
// Created by steinraf on 20/12/22.
//

#pragma once

#include "../cudaHelpers.cuh"
#include "../utility/vector.cuh"


__global__ void denoise(Vector3f *input, Vector3f *output, FeatureBuffer featureBuffer, float *weights, int width, int height,
                        Vector3f cameraOrigin = Vector3f{0.f});

__global__ void denoiseApplyWeights(Vector3f *output, float *weights, int width, int height);

GPU_ONLY void bilateralFilterSlides(Vector3f *input, Vector3f *output, FeatureBuffer &featureBuffer, float *weights,  int i, int j, int width, int height);
