//
// Created by steinraf on 13.03.25.
//

#pragma once

#include <utility>

#include "../common.cuh"

#include "../geometry/ray.cuh"

#include "../acceleration/multibvh.cuh"

class AreaLight {
public:
    explicit AreaLight(const EmitterDescriptorHost &emitterDescriptor) noexcept;

    [[nodiscard]] __device__ AABB getBoundingBox() const noexcept;

    [[nodiscard]] __device__ bool intersect(const Ray &ray, Intersection &its, bool isShadowRay = false) const noexcept;

    [[nodiscard]] __device__ float pdf(const EmitterQueryRecord &eqr) const noexcept;
    [[nodiscard]] __device__ Color eval(const EmitterQueryRecord &eqr) const noexcept;
    [[nodiscard]] __device__ Color sample(EmitterQueryRecord &eqr, const Vec3f &rng) const;

    const BLAS *blas;

    Color radiance;
};

struct EmitterQueryRecord {
    Vec3f ref;    // reference point
    Vec3f point;  // intersection point
    Vec3f normal; // surface normal
    Vec3f wIn;    // incident light omega
    Vec2f uv;     // emitter uv
    float pdf;    // emitter pdf
    size_t idx;   // emitter cdf index
    Ray shadowRay;// shadow ray

    __host__ __device__ explicit EmitterQueryRecord(Vec3f ref) noexcept
        : ref(std::move(ref)), point(), normal(), wIn(), uv(), pdf(), idx(), shadowRay({0.0f, 0.0f, 0.0f}, {0.0f, 0.0f, -1.0f}) {
    }

    __host__ __device__ EmitterQueryRecord(Vec3f ref, Vec3f p, Vec3f n, Vec2f uv) noexcept
        : ref(std::move(ref)), point(std::move(p)), normal(std::move(n)), wIn((p - ref).normalized()), uv(std::move(uv)), pdf(), idx(), shadowRay({0.0f, 0.0f, 0.0f}, {0.0f, 0.0f, -1.0f}) {
    }
};
