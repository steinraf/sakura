//
// Created by steinraf on 13.03.25.
//

#pragma once

#include "../common.cuh"

#include "../acceleration/multibvh.cuh"

class AreaLight {
public:
    explicit AreaLight(const EmitterDescriptorHost &emitterDescriptor) noexcept;

    [[nodiscard]] __device__ AABB getBoundingBox() const noexcept;

    [[nodiscard]] __device__ bool intersect(const Ray &ray, Intersection &its, bool isShadowRay = false) const noexcept;


    const BLAS *blas;

    Color radiance;
};

struct EmitterQueryRecord {
    Vec3f point;
};
