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
    __host__ __device__ Triangle(
            Eigen::Vector3f p0, Eigen::Vector3f p1, Eigen::Vector3f p2,
            const Eigen::Vector3f &n0, const Eigen::Vector3f &n1, const Eigen::Vector3f &n2,
            Eigen::Vector2f uv0, Eigen::Vector2f uv1, Eigen::Vector2f uv2) noexcept;

    __device__ Triangle() noexcept;

    __host__ __device__ bool intersectHandler(const Ray &ray,
                                              Intersection &its) const noexcept;

    // TODO check if it makes sense to store as a field
    __host__ __device__ AABB AABBGetter() const noexcept;

    __host__ __device__ void hitInformationSetter(
            const Ray &ray, Intersection &its) const noexcept;

    __host__ __device__ float getArea() const noexcept;

    [[nodiscard]] __host__ __device__ Eigen::Vector3f getCoordinate(const Vec3f &c /* barycentric coordinates */) const noexcept;
    [[nodiscard]] __host__ __device__ Eigen::Vector3f getNormal(const Vec3f &c /* barycentric coordinates */) const noexcept;
    [[nodiscard]] __host__ __device__ Eigen::Vector2f getUV(const Vec3f &c /* barycentric coordinates */) const noexcept;

private:
    Eigen::Vector3f p0, p1, p2;
    Eigen::Vector3f n0, n1, n2;
    Eigen::Vector2f uv0, uv1, uv2;
};

struct ShapeQueryRecord {
    Vec3f ref;
    Vec3f point;
    Vec3f normal;
    Vec2f uv;
    float pdf;

    __host__ __device__ explicit ShapeQueryRecord(Vec3f ref) noexcept
        : ref(std::move(ref)), point(), normal(), uv(), pdf() {
    }

    __host__ __device__ ShapeQueryRecord(Vec3f ref, Vec3f p) noexcept
        : ref(std::move(ref)), point(std::move(p)), normal(), uv(), pdf() {
    }
};