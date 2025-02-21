//
// Created by steinraf on 15.02.25.
//

#pragma once

#include "../common.cuh"

__global__ void denoise(cudaSurfaceObject_t surface, const FeatureBuffer *buffer, int width, int height);