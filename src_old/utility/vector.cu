//
// Created by steinraf on 05.04.25.
//

#include "vector.cuh"

__device__ void Vector3f::atomicCudaAdd(Vector3f *address, const Vector3f &vec) noexcept {
    Vector3f &v = *address;
    atomicAdd(&(v[0]), vec[0]);
    atomicAdd(&(v[1]), vec[1]);
    atomicAdd(&(v[2]), vec[2]);
}

