//
// Created by steinraf on 04.02.25.
//

#pragma once

#include <Eigen/Dense>
#include <utility>

#include "../common.cuh"

static const float TRIANGLE_COLLISION_EPSILON = 0.0001f;


class Triangle {
public:
    CPU_GPU Triangle(
            Vec3f p0, Vec3f p1, Vec3f p2,
            const Vec3f &n0, const Vec3f &n1, const Vec3f &n2,
            Vec2f uv0, Vec2f uv1, Vec2f uv2) noexcept;

    __device__ Triangle() noexcept;

    CPU_GPU bool intersectHandler(const Ray &ray,
                                  Intersection &its) const noexcept;

    // TODO check if it makes sense to store as a field
    CPU_GPU AABB AABBGetter() const noexcept;

    CPU_GPU void hitInformationSetter(
            const Ray &ray, Intersection &its) const noexcept;

    CPU_GPU float getArea() const noexcept;

    [[nodiscard]] CPU_GPU Vec3f getCoordinate(const Vec3f &c /* barycentric coordinates */) const noexcept;
    [[nodiscard]] CPU_GPU Vec3f getNormal(const Vec3f &c /* barycentric coordinates */) const noexcept;
    [[nodiscard]] CPU_GPU Vec2f getUV(const Vec3f &c /* barycentric coordinates */) const noexcept;

private:
    Vec3f p0, p1, p2;
    Vec3f n0, n1, n2;
    Vec2f uv0, uv1, uv2;
};

struct ShapeQueryRecord {
    Vec3f ref;
    Vec3f point;
    Vec3f normal;
    Vec2f uv;
    float pdf;

    CPU_GPU explicit ShapeQueryRecord(Vec3f ref) noexcept
        : ref(std::move(ref)), point(), normal(), uv(), pdf() {
    }

    CPU_GPU ShapeQueryRecord(Vec3f ref, Vec3f p) noexcept
        : ref(std::move(ref)), point(std::move(p)), normal(), uv(), pdf() {
    }
};