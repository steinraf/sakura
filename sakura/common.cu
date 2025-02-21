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
