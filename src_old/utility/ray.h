//
// Created by steinraf on 19/08/22.
//

#pragma once

#include "../../src/common.h"

#include <cuda/std/limits>


class Ray3f {
public:
    [[nodiscard]] CPU_GPU_CONSTEXPR Ray3f() noexcept
        : o{0.f, 0.f, 0.f}, minDist(EPSILON), d{1.f, 0.f, 0.f},
          maxDist(std::numeric_limits<float>::infinity()) {
    }

    [[nodiscard]] CPU_GPU_CONSTEXPR Ray3f(const Vector3f &origin, const Vector3f &direction,
                                             float minDist = EPSILON,
                                             float maxDist = cuda::std::numeric_limits<float>::infinity()) noexcept
        : o(origin), minDist(minDist), d(direction), maxDist(maxDist) {
    }

    [[nodiscard]] CPU_GPU_CONSTEXPR Vector3f atTime(float t) const noexcept { return o + t * d; }

    [[nodiscard]] CPU_GPU_CONSTEXPR Vector3f getOrigin() const noexcept { return o; }

    [[nodiscard]] CPU_GPU_CONSTEXPR Vector3f getDirection() const noexcept { return d; }

    [[nodiscard]] CPU_GPU_CONSTEXPR float getMinDist() const noexcept { return minDist; }
    [[nodiscard]] CPU_GPU_CONSTEXPR float getMaxDist() const noexcept { return maxDist; }

    CPU_GPU_CONSTEXPR void setMinDist(float max_dist) noexcept { this->minDist = max_dist; }
    CPU_GPU_CONSTEXPR void setMaxDist(float min_dist) noexcept { this->maxDist = min_dist; }


private:
    Vec3f o;
    float minDist;
    Vec3f d;
    float maxDist;
};
