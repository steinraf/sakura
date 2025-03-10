//
// Created by steinraf on 25.02.25.
//

#pragma once

#include "../common.cuh"
#include "../geometry/triangle.cuh"
#include "../material/material.cuh"
#include "../texture/texture.cuh"

struct MeshDescriptorHost {
    std::vector<Triangle> triangles;
    Eigen::Affine3f transform;
    Material material;
    Texture texture;
};

class BLAS {
public:
    __host__ explicit BLAS(const std::vector<Triangle> &triangles, const Eigen::Affine3f &transform) noexcept;
    // BVH needs to be GPU accessible memory
    __host__ ~BLAS();

    __host__ __device__ BLAS(const BLAS &other) noexcept = delete;
    __host__ __device__ BLAS &operator=(const BLAS &other) noexcept = delete;
    __host__ __device__ BLAS(BLAS &&other) noexcept = delete;
    __host__ __device__ BLAS &operator=(BLAS &&other) noexcept = default;


    [[nodiscard]] __device__ bool intersect(
            Ray ray, Intersection &its,
            bool isShadowRay = false) const noexcept;

private:
    BVH *bvh;
    Eigen::Affine3f inverseTransform;
};


class TLAS {
public:
    __host__ explicit TLAS(const std::vector<MeshDescriptorHost> &blases) noexcept;


    [[nodiscard]] __device__ bool intersect(
            const Ray &ray, Intersection &its,
            bool isShadowRay = false) const noexcept;

    void __host__ cleanup() {
        if(blas) return;

        checkCudaErrors(cudaFree(blas));
    }

private:
    BLAS *blas;
    size_t numBlas;
};