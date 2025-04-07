//
// Created by steinraf on 07.04.25.
//

#pragma once

#include "../src_old/utility/vector.cuh"

struct EmitterQueryRecord;
class Ray3f;
class Scene;
struct ShapeQueryRecord;





using Vec2f = Vector2f;
using Vec3f = Vector3f;

#define CPU_GPU __host__ __device__
#define CPU_GPU_CONSTEXPR __host__ __device__ inline constexpr
#define CPU_ONLY __host__
#define GPU_ONLY __device__

//
//void checkCudaErrors(cudaError result) {
//    if(result != cudaSuccess) {
//        fprintf(stderr, "CUDA Runtime Error: %s\n", cudaGetErrorString(result));
//        exit(-1);
//    }
//}