//
// Created by steinraf on 25.02.25.
//

#pragma once

#include "../common.cuh"
#include "../geometry/triangle.cuh"
#include "../material/bsdf.cuh"
#include "../rng/sampler.cuh"
#include "aabb.cuh"

struct MeshDescriptorHost {
    std::vector<Triangle> triangles;
    Eigen::Affine3f transform;
    BSDF bsdf;
};

struct EmitterDescriptorHost {
    std::vector<Triangle> triangles;
    Eigen::Affine3f transform;
    BSDF bsdf;
    Vec3f radiance;
};

class BLAS {
public:
    __host__ explicit BLAS(const MeshDescriptorHost &meshDescriptor) noexcept;

    [[nodiscard]] __host__ __device__ AABB getBoundingBox() const noexcept;// todo account for transform

    [[nodiscard]] __device__ bool intersect(
            Ray ray, Intersection &its,
            bool isShadowRay = false) const noexcept;

    [[nodiscard]] __host__ __device__ float pdfSurface(const ShapeQueryRecord &sqr) const noexcept;

    __host__ __device__ void sampleSurface(ShapeQueryRecord &sqr, const Vec3f &rng) const noexcept;

    BSDF bsdf;

private:
    BVH *bvh;

    Eigen::Affine3f transform;
    Eigen::Affine3f inverseTransform;

    friend TLAS;
};


class TLAS {
public:
    __host__ explicit TLAS(const std::vector<MeshDescriptorHost> &meshes, const std::vector<EmitterDescriptorHost> &emitters) noexcept;


    [[nodiscard]] __device__ bool intersect(
            const Ray &ray, Intersection &its,
            bool isShadowRay = false) const noexcept;

    [[nodiscard]] __device__ bool intersect(const Ray &ray) const noexcept;

    void __host__ cleanup();

    __host__ __device__ AABB getBoundingBox() const {
        return boundingBox;
    }


    __host__ __device__ const AreaLight *getRandomEmitter(float d);
    __host__ __device__ size_t getEmitterCount() const {
        return numEmitters;
    }
    __host__ __device__ bool containsEmitters() const {
        return numEmitters > 0;
    }

private:
    BLAS *meshes;
    size_t numMeshes;

    AreaLight *emitters;
    size_t numEmitters;

    AABB boundingBox;// todo account for transform
};