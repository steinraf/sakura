//
// Created by steinraf on 05.02.25.
//

#pragma once

#include "../acceleration/bvh.cuh"
#include "../camera/camera.cuh"
#include "../gui/gui.cuh"


__global__ void render_kern(BVH *bvh, struct FeatureBuffer *buffer, Camera camera, curandState *rngStates, int width, int height, int spp);

__global__ void bufferToSurface(cudaSurfaceObject_t surface, FeatureBuffer *buffer, int width, int height);
