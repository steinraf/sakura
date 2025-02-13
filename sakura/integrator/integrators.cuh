//
// Created by steinraf on 05.02.25.
//

#pragma once

#include "../acceleration/bvh.cuh"
#include "../camera/camera.cuh"


__global__ void render_kern(BVH *bvh, cudaSurfaceObject_t surface, Camera camera, curandState *rngStates, int width, int height, int spp);
