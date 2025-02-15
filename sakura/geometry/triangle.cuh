//
// Created by steinraf on 04.02.25.
//

#pragma once

#include "../acceleration/aabb.cuh"
#include "intersection.cuh"

static const float TRIANGLE_COLLISION_EPSILON = 0.0001f;

class Triangle {
public:
    __host__ __device__ Triangle(
            Eigen::Vector3f p0, Eigen::Vector3f p1, Eigen::Vector3f p2,
            const Eigen::Vector3f &n0, const Eigen::Vector3f &n1,
            const Eigen::Vector3f &n2
            //             const Eigen::Vector2f& uv0, const Eigen::Vector2f&
            //             uv1, const Eigen::Vector2f& uv2
            ) noexcept;

    __device__ Triangle() noexcept;

    __host__ __device__ bool intersectHandler(const Ray &ray,
                                              Intersection &its) const noexcept;

    // TODO check if it makes sense to store as a field
    __host__ __device__ AABB AABBGetter() const noexcept;

    __host__ __device__ void hitInformationSetter(
            const Ray &ray, Intersection &its) const noexcept;

    Eigen::Vector3f p0, p1, p2;
    Eigen::Vector3f n0, n1, n2;
    //    Eigen::Vector2f uv0, uv1, uv2;
};