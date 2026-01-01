//
// Created by steinraf on 04.02.25.
//
#include <utility>

#include "ray.cuh"

CPU_GPU Ray::Ray(Vec3f origin, Vec3f dir, float minDist, float maxDist) noexcept
    : origin(std::move(origin)), dir(std::move(dir)), minDist(minDist), maxDist(maxDist) {
}
CPU_GPU Vec3f Ray::at(float t) const noexcept {
    assert(t >= minDist && t <= maxDist);
    return origin + t * dir;
}
CPU_GPU void Ray::transform(const Eigen::Affine3f &transform) noexcept {
    origin = transform * origin;
    dir = transform.linear() * dir;// No normalization because t should not be affected
}
