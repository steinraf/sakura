//
// Created by steinraf on 05.02.25.
//

#pragma once

#include <curand_kernel.h>

#include <Eigen/Dense>

#include "../common.cuh"


//TODO have camera object live on the device to save copies
__global__ void render_kern(TLAS *tlas, Texture envmap, FeatureBuffer *buffer, Camera camera, curandState *rngStates, unsigned int width, unsigned int height, int spp);

__global__ void bufferToSurface(cudaSurfaceObject_t surface, FeatureBuffer *buffer, unsigned int width, unsigned int height);


__device__ Eigen::Vector3f tonemap(Eigen::Vector3f color);