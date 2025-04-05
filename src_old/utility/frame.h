//
// Created by steinraf on 29/11/22.
//

#pragma once

#include "vector.cuh"

//Nori Frame
class Frame {
private:
    Vector3f s, t;

public:
    Vector3f n;

    constexpr Frame() = default;

    __device__ constexpr explicit Frame(const Vector3f &nIn) noexcept
        : n(nIn.normalized()) {
        if(abs(n[0]) > abs(n[1])) {
            const float invLen = 1.0f / sqrt(n[0] * n[0] + n[2] * n[2]);
            t = Vector3f(n[2] * invLen, 0.0f, -n[0] * invLen).normalized();
        } else {
            const float invLen = 1.0f / std::sqrt(n[1] * n[1] + n[2] * n[2]);
            t = Vector3f(0.0f, n[2] * invLen, -n[1] * invLen).normalized();
        }
        s = t.cross(n).normalized();
    }

    __device__ constexpr Frame(const Vector3f &s, const Vector3f &t, const Vector3f &n) noexcept
            :s(s.normalized()), t(t.normalized()), n(n.normalized()){

    }

    __device__ constexpr Vector3f toLocal(const Vector3f &v) const noexcept {
        return {v.dot(s), v.dot(t), v.dot(n)};
    }

    __device__ constexpr Vector3f toWorld(const Vector3f &v) const noexcept {
        return s * v[0] + t * v[1] + n * v[2];
    }

    __device__ static constexpr inline float cosTheta(const Vector3f &v) noexcept {
        return v[2];
    }
};
