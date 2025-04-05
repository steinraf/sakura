//
// Created by steinraf on 28/11/22.
//

#pragma once

#include "../shapes/triangle.h"
#include "../utility/ray.h"
#include "../utility/vector.cuh"

struct EmitterQueryRecord {
    Vector3f ref;
    Vector3f p;
    Vector3f n;
    Vector3f wi;
    Vector2f uv;
    float pdf;
    size_t idx;
    Ray3f shadowRay;

    __device__ constexpr explicit EmitterQueryRecord(const Vector3f &ref) noexcept
        : ref(ref), p(), n(), wi(), uv(), pdf(), idx(), shadowRay() {
    }

    __device__ constexpr EmitterQueryRecord(const Vector3f &ref, const Vector3f &p, const Vector3f &n, const Vector2f &uv) noexcept
        : ref(ref), p(p), n(n), wi((p - ref).normalized()), uv(uv), pdf(), idx(), shadowRay() {
    }
};


class AreaLight {
public:
    explicit __host__ __device__ constexpr AreaLight(const Color3f &radiance) noexcept
        : radiance(radiance), blas(nullptr) {
    }

    AreaLight() = default;

    __device__ void constexpr setBlas(const class BLAS *newBlas) {
        blas = newBlas;
    }


    [[nodiscard]] __device__ Color3f eval(const EmitterQueryRecord &emitterQueryRecord) const noexcept;

    [[nodiscard]] __device__ float pdf(const EmitterQueryRecord &emitterQueryRecord) const noexcept;


    [[nodiscard]] __device__ Color3f
    sample(EmitterQueryRecord &emitterQueryRecord, const Vector3f &sample) const noexcept;


    [[nodiscard]] __device__ constexpr bool isEmitter() const noexcept {
        return !radiance.isZero();
    }

    //TODO add texture as radiance option
    Color3f radiance;
    const BLAS *blas = nullptr;
};