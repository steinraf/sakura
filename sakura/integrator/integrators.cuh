//
// Created by steinraf on 05.02.25.
//

#pragma once

#include <curand_kernel.h>

#include <Eigen/Dense>

#include "../common.cuh"


__global__ void render_kern(BVH *bvh, struct FeatureBuffer *buffer, Camera camera, curandState *rngStates, int width, int height, int spp);

__global__ void bufferToSurface(cudaSurfaceObject_t surface, FeatureBuffer *buffer, int width, int height);


__device__ Eigen::Vector3f tonemap(Eigen::Vector3f color);