//
// Created by steinraf on 04.02.25.
//

#pragma once
#include <Eigen/Dense>

struct Intersection {
    Eigen::Vector3f point;
    Eigen::Vector3f normal;

    Eigen::Vector2f uv;
    float t;

    const class Triangle *triangle;
    const class Mesh *mesh;
};

// Taken from Nori
class Frame {
private:
    Eigen::Vector3f s, t;

public:
    Eigen::Vector3f n;

    __device__ explicit Frame(const Eigen::Vector3f &nIn) noexcept
        : n(nIn.normalized()) {
        if(abs(n[0]) > abs(n[1])) {
            assert(sqrt(n[0] * n[0] + n[2] * n[2]) > 0);
            const float invLen = 1.0f / sqrt(n[0] * n[0] + n[2] * n[2]);
            t = Eigen::Vector3f(n[2] * invLen, 0.0f, -n[0] * invLen)
                        .normalized();
        } else {
            assert(std::sqrt(n[1] * n[1] + n[2] * n[2]) > 0);
            const float invLen = 1.0f / std::sqrt(n[1] * n[1] + n[2] * n[2]);
            t = Eigen::Vector3f(0.0f, n[2] * invLen, -n[1] * invLen)
                        .normalized();
        }
        s = t.cross(n).normalized();
    }

    __device__ Frame(const Eigen::Vector3f &s, const Eigen::Vector3f &t,
                     const Eigen::Vector3f &n) noexcept
        : s(s.normalized()), t(t.normalized()), n(n.normalized()) {}

    __device__ Eigen::Vector3f toLocal(
            const Eigen::Vector3f &v) const noexcept {
        return {v.dot(s), v.dot(t), v.dot(n)};
    }

    __device__ Eigen::Vector3f toWorld(
            const Eigen::Vector3f &v) const noexcept {
        return s * v[0] + t * v[1] + n * v[2];
    }

    __device__ static inline float cosTheta(const Eigen::Vector3f &v) noexcept {
        return v[2];
    }
};