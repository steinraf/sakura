//
// Created by steinraf on 04.02.25.
//
#include <utility>

#include "ray.cuh"

__host__ __device__ Ray::Ray(Eigen::Vector3f origin, Eigen::Vector3f dir, float minDist, float maxDist) noexcept
    : origin(std::move(origin)), dir(std::move(dir)), minDist(minDist), maxDist(maxDist) {
}
__host__ __device__ Eigen::Vector3f Ray::at(float t) const noexcept {
    assert(t >= minDist && t <= maxDist);
    return origin + t * dir;
}
__host__ __device__ void Ray::transform(const Eigen::Matrix4f &transform) noexcept {
    origin = (transform * Eigen::Vector4f(origin.x(), origin.y(), origin.z(), 1)).head<3>();
    dir = (transform * Eigen::Vector4f(dir.x(), dir.y(), dir.z(), 0)).head<3>();
    dir.normalize();
}
