//
// Created by steinraf on 23/10/22.
//

#pragma once

#include "vector.cuh"
#include <curand_kernel.h>
#include <random>

struct HostRNG {
    __host__ HostRNG() : rd(), gen(rd()), dis(0.0f, 1.0f) {
    }

    [[nodiscard]] __host__ float getSample() {return dis(gen);}

    std::random_device rd;
    std::mt19937 gen;
    std::uniform_real_distribution<float> dis;
};

class Sampler {
public:
    CPU_GPU_INLINE explicit Sampler(curandState *curand)
#ifdef __CUDA_ARCH__
        : rng(curand) {
#else
        : hostRNG() {

#endif
    }

    CPU_GPU_INLINE ~Sampler(){};


    [[nodiscard]] CPU_GPU_INLINE float getSample1D() {
#ifdef __CUDA_ARCH__
        return curand_uniform(rng);
#else
        return hostRNG.getSample();
#endif
    }


    [[nodiscard]] CPU_GPU_INLINE Vector2f getSample2D() {
#ifdef __CUDA_ARCH__
        return {curand_uniform(rng), curand_uniform(rng)};
#else
        return {hostRNG.getSample(), hostRNG.getSample()};
#endif
    }

    [[nodiscard]] CPU_GPU_INLINE Vector3f getSample3D() {

#ifdef __CUDA_ARCH__
        return {curand_uniform(rng), curand_uniform(rng), curand_uniform(rng)};
#else
        return {hostRNG.getSample(), hostRNG.getSample(), hostRNG.getSample()};
#endif
    }

private:
    union {
        curandState *rng;
        HostRNG hostRNG;
    };

};
