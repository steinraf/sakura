//
// Created by steinraf on 04.02.25.
//

#pragma once
#include "../common.cuh"
#include <Eigen/Dense>


// Taken from Nori
class Frame {
private:
    Vec3f s, t;

public:
    Vec3f n;

    CPU_GPU Frame() noexcept
        : Frame({0.0, 1.0, 0.0}) {}

    CPU_GPU explicit Frame(const Vec3f &nIn) noexcept
        : n(nIn.normalized()) {
        if(abs(n[0]) > abs(n[1])) {
            assert(sqrt(n[0] * n[0] + n[2] * n[2]) > 0);
            const float invLen = 1.0f / sqrt(n[0] * n[0] + n[2] * n[2]);
            t = Vec3f(n[2] * invLen, 0.0f, -n[0] * invLen)
                        .normalized();
        } else {
            assert(std::sqrt(n[1] * n[1] + n[2] * n[2]) > 0);
            const float invLen = 1.0f / std::sqrt(n[1] * n[1] + n[2] * n[2]);
            t = Vec3f(0.0f, n[2] * invLen, -n[1] * invLen)
                        .normalized();
        }
        s = t.cross(n).normalized();
    }

    CPU_GPU Frame(const Vec3f &s, const Vec3f &t,
                  const Vec3f &n) noexcept
        : s(s.normalized()), t(t.normalized()), n(n.normalized()) {}

    CPU_GPU Vec3f toLocal(
            const Vec3f &v) const noexcept {
        return {v.dot(s), v.dot(t), v.dot(n)};
    }

    CPU_GPU Vec3f toWorld(
            const Vec3f &v) const noexcept {
        //        assert(std::isfinite(v[0]));
        //        assert(std::isfinite(v[1]));
        //        assert(std::isfinite(v[2]));
        return s * v[0] + t * v[1] + n * v[2];
    }

    CPU_GPU static inline float cosTheta(const Vec3f &v) noexcept {
        return v[2];
    }

    CPU_GPU static inline float tanTheta(const Vec3f &v) noexcept {
        float k = 1 - v[2] * v[2];
        if(k < 0)
            return 0;
        return std::sqrt(k) / v[2];
    }

    CPU_GPU void rotate(const Eigen::Affine3f &tf) {
        *this = Frame(tf.linear() * n);
    }
};


struct Intersection {
    Vec3f point;
    Frame shFrame;// Shading frame

    Vec2f uv;
    float t;

    const Triangle *triangle;
    const BLAS *meshf;
    const AreaLight *emitter;

    CPU_GPU bool isEmitter() const {
        return emitter != nullptr;
    }
};