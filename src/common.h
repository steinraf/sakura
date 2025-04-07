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

#define CPU_GPU __host__ __device__
#define CPU_GPU_INLINE __host__ __device__ inline
#define CPU_GPU_CONSTEXPR __host__ __device__ inline constexpr
#define CPU_ONLY __host__
#define GPU_ONLY __device__


__host__ inline void check_cuda(cudaError_t result, char const *const func, const char *const file, int line) noexcept(false) {
    if(result) {
        std::cerr << "CUDA error = " << static_cast<unsigned int>(result) << ":" << cudaGetErrorString(result) << "\nAt " << file << ":" << line << " '" << func << "' \n";
        cudaDeviceReset();
        exit(99);
    }
}

#define checkCudaErrors(val) check_cuda((val), #val, __FILE__, __LINE__)

