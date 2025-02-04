//
// Created by steinraf on 04.02.25.
//

#pragma once

#include "intersection.cuh"
#include "ray.cuh"

static const float TRIANGLE_COLLISION_EPSILON = 0.0001f;

class Triangle{
public:

    Eigen::Vector3f p0, p1, p2; // Vertex positions
    Eigen::Vector3f n0, n1, n2; // Vertex normals (normalized)


    __host__ __device__ Triangle(Eigen::Vector3f  p0, Eigen::Vector3f  p1, Eigen::Vector3f  p2,
                                 const Eigen::Vector3f& n0, const Eigen::Vector3f& n1, const Eigen::Vector3f& n2 ) noexcept;


    // Nori ray intersection code
    __host__ __device__ bool intersect(const Ray& ray, Intersection& its) const noexcept;
};