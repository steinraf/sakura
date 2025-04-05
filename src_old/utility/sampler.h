//
// Created by steinraf on 23/10/22.
//

#pragma once

#include "vector.cuh"
#include <curand_kernel.h>

class Sampler {
public:
    __device__ explicit Sampler(curandState *curand) : rng(curand) {}

    [[nodiscard]] __device__ float getSample1D() { return curand_uniform(rng); }

    [[nodiscard]] __device__ Vector2f getSample2D() { return {curand_uniform(rng), curand_uniform(rng)}; }

    [[nodiscard]] __device__ Vector3f getSample3D() {
        return {curand_uniform(rng), curand_uniform(rng), curand_uniform(rng)};
    }

private:
    curandState *rng;
};
