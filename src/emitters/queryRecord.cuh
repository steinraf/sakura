//
// Created by steinraf on 07.04.25.
//

#pragma once

#include "../../src_old/utility/ray.h"
#include "../common.h"


struct EmitterQueryRecord {
    Vec3f ref;
    Vec3f p;
    Vec3f n;
    Vec3f wi;
    Vec2f uv;
    float pdf;
    size_t idx;
    Ray3f shadowRay;

    CPU_GPU_CONSTEXPR explicit EmitterQueryRecord(const Vector3f &ref) noexcept
        : ref(ref), p(), n(), wi(), uv(), pdf(), idx(), shadowRay() {
    }

    CPU_GPU_CONSTEXPR EmitterQueryRecord(const Vector3f &ref, const Vec3f &p, const Vec3f &n, const Vec2f &uv) noexcept
        : ref(ref), p(p), n(n), wi((p - ref).normalized()), uv(uv), pdf(), idx(), shadowRay() {
    }
};