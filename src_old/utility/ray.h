//
// Created by steinraf on 19/08/22.
//

#pragma once

#include "../../src/common.h"

#include <cuda/std/limits>


class Ray3f {
public:
    [[nodiscard]] CPU_GPU_CONSTEXPR Ray3f() noexcept
        : o{0.f, 0.f, 0.f}, d{1.f, 0.f, 0.f}, minDist(EPSILON),
          maxDist(std::numeric_limits<float>::infinity()) {
    }

    [[nodiscard]] CPU_GPU_CONSTEXPR Ray3f(const Vector3f &origin, const Vector3f &direction,
                                             float minDist = EPSILON,
                                             float maxDist = cuda::std::numeric_limits<float>::infinity()) noexcept
        : o(origin), d(direction), minDist(minDist), maxDist(maxDist) {
    }

    [[nodiscard]] CPU_GPU_CONSTEXPR Vector3f atTime(float t) const noexcept { return o + t * d; }

    [[nodiscard]] CPU_GPU_CONSTEXPR Vector3f getOrigin() const noexcept { return o; }

    [[nodiscard]] CPU_GPU_CONSTEXPR Vector3f getDirection() const noexcept { return d; }


    Vec3f o;
    Vec3f d;
    float minDist;
    float maxDist;
};
