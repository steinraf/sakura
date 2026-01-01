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

    [[nodiscard]] CPU_GPU AABB getBoundingBox() const noexcept;// todo account for transform

    [[nodiscard]] __device__ bool intersect(
            Ray ray, Intersection &its,
            bool isShadowRay = false) const noexcept;

    [[nodiscard]] CPU_GPU bool intersectAABB(Ray ray, Intersection &its) const noexcept;

    [[nodiscard]] CPU_GPU float pdfSurface(const ShapeQueryRecord &sqr) const noexcept;

    CPU_GPU void sampleSurface(ShapeQueryRecord &sqr, const Vec3f &rng) const noexcept;

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

    [[nodiscard]] __device__ bool intersectAABB(const Ray &ray, Intersection &its) const noexcept;

    [[nodiscard]] __device__ bool intersect(const Ray &ray) const noexcept;

    void __host__ cleanup();

    CPU_GPU AABB getBoundingBox() const {
        return boundingBox;
    }


    CPU_GPU const AreaLight *getRandomEmitter(float d);
    CPU_GPU size_t getEmitterCount() const {
        return numEmitters;
    }
    CPU_GPU bool containsEmitters() const {
        return numEmitters > 0;
    }

private:
    BLAS *meshes;
    size_t numMeshes;

    AreaLight *emitters;
    size_t numEmitters;

    AABB boundingBox;// todo account for transform
};