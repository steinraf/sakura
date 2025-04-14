//
// Created by steinraf on 07.04.25.
//



#pragma once

#include "../src_old/utility/vector.cuh"

class AreaLight;
class BLAS;
struct EmitterQueryRecord;
class Ray3f;
class Scene;
struct ShapeQueryRecord;

using Vec2f = Vector2f;
using Vec3f = Vector3f;




__host__ inline void check_cuda(cudaError_t result, char const *const func, const char *const file, int line) noexcept(false) {
    if(result) {
        std::cerr << "CUDA error = " << static_cast<unsigned int>(result) << ":" << cudaGetErrorString(result) << "\nAt " << file << ":" << line << " '" << func << "' \n";
        cudaDeviceReset();
        throw std::runtime_error("CUDA error");
    }
}

#define checkCudaErrors(val) check_cuda((val), #val, __FILE__, __LINE__)

