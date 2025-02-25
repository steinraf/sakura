//
// Created by steinraf on 17.02.25.
//


#include <iostream>

#include "common.cuh"

void checkCudaErrors(cudaError result) {
    if(result != cudaSuccess) {
        fprintf(stderr, "CUDA Runtime Error: %s\n", cudaGetErrorString(result));
        exit(-1);
    }
}


__host__ __device__ int safe_uint_to_int(unsigned int i) {
    if(i > static_cast<unsigned int>(std::numeric_limits<int>::max())) {
        // If we compile in debug, throw an assertion
        // Otherwise, clamp to int max
#ifndef NDEBUG
        assert(!"Unsigned int cannot be converted to int");
#else
        return std::numeric_limits<int>::max();
#endif
    }
    return static_cast<int>(i);
}