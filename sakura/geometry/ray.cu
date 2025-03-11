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
__host__ __device__ void Ray::transform(const Eigen::Affine3f &transform) noexcept {
    origin = transform * origin;
    dir = transform.linear() * dir;// No normalization because t should not be affected
}
