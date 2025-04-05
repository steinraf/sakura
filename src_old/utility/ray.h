//
// Created by steinraf on 19/08/22.
//

#pragma once

#include "vector.cuh"
#include <cuda/std/limits>


class Ray3f {
public:
    [[nodiscard]] __host__ __device__ constexpr Ray3f()
        : o{0.f, 0.f, 0.f}, d{1.f, 0.f, 0.f}, minDist(EPSILON),
          maxDist(cuda::std::numeric_limits<float>::infinity()) {
    }

    [[nodiscard]] __device__ constexpr Ray3f(const Vector3f &origin, const Vector3f &direction,
                                             float minDist = EPSILON,
                                             float maxDist = cuda::std::numeric_limits<float>::infinity()) noexcept
        : o(origin), d(direction), minDist(minDist), maxDist(maxDist) {
    }

    [[nodiscard]] __device__ constexpr Vector3f atTime(float t) const noexcept { return o + t * d; }

    [[nodiscard]] __device__ constexpr Vector3f getOrigin() const noexcept { return o; }

    [[nodiscard]] __device__ constexpr Vector3f getDirection() const noexcept { return d; }


    Vector3f o;
    Vector3f d;
    float minDist;
    float maxDist;
};
