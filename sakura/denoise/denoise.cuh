//
// Created by steinraf on 15.02.25.
//

#pragma once

#include "../common.cuh"

__global__ void denoise(const FeatureBuffer *bufferIn, Vec3f *output, float *weights, unsigned int width, unsigned int height);
__global__ void applyWeights(cudaSurfaceObject_t surface, const Vec3f *buffer, float *weights, unsigned int width, unsigned int height);