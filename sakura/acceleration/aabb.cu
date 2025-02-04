//
// Created by steinraf on 04.02.25.
//

#include <utility>

#include "aabb.cuh"

__host__ __device__ AABB::AABB() noexcept
    : min{cuda::std::numeric_limits<float>::max(),
          cuda::std::numeric_limits<float>::max(),
          cuda::std::numeric_limits<float>::max()},
      max{cuda::std::numeric_limits<float>::lowest(),
          cuda::std::numeric_limits<float>::lowest(),
          cuda::std::numeric_limits<float>::lowest()} {}

__host__ __device__ AABB::AABB(Eigen::Vector3f min,
                               Eigen::Vector3f max) noexcept
    : min{std::move(min)}, max{std::move(max)} {
#ifndef NDEBUG
    if (isFaulty()) {
        printf("AABB IS EMPTY (%f %f %f %f %f %f)\n", min.x(), min.y(), min.z(),
               max.x(), max.y(), max.z());
        assert(!"EmptyAABB CONSTRUCTED");
    }
    assert(!isFaulty());
#endif
}

__host__ __device__ AABB::AABB(const Eigen::Vector3f &p0,
                               const Eigen::Vector3f &p1,
                               const Eigen::Vector3f &p2) noexcept
    : min{thrust::min(thrust::min(p0.x(), p1.x()), p2.x()),
          thrust::min(thrust::min(p0.y(), p1.y()), p2.y()),
          thrust::min(thrust::min(p0.z(), p1.z()), p2.z())},
      max{thrust::max(thrust::max(p0.x(), p1.x()), p2.x()),
          thrust::max(thrust::max(p0.y(), p1.y()), p2.y()),
          thrust::max(thrust::max(p0.z(), p1.z()), p2.z())} {
#ifndef NDEBUG
    if (isFaulty()) {
        printf("AABB IS EMPTY (%f %f %f %f %f %f)\n", min.x(), min.y(), min.z(),
               max.x(), max.y(), max.z());
        assert(!"EmptyAABB CONSTRUCTED");
    }
    assert(!isFaulty());
#endif
}

[[nodiscard]] __host__ __device__ bool AABB::intersect(
        const Ray &ray) const noexcept {
    float nearT = cuda::std::numeric_limits<float>::lowest();
    float farT = cuda::std::numeric_limits<float>::max();

    for (int i = 0; i < 3; i++) {
        float origin = ray.origin[i];
        float minVal = min[i], maxVal = max[i];

        if (ray.dir[i] == 0) {
            if (origin < minVal || origin > maxVal) return false;
        } else {
            float t1 = (minVal - origin) / ray.dir[i];
            float t2 = (maxVal - origin) / ray.dir[i];

            if (t1 > t2) {
                cuda::std::swap(t1, t2);
            }

            nearT = thrust::max(t1, nearT);
            farT = thrust::min(t2, farT);

            if (nearT > farT) return false;
        }
    }

    return ray.minDist <= farT && nearT <= ray.maxDist;
}

// TODO check if adding -+ Epsilon to bounds is necessary
[[nodiscard]] __host__ __device__ AABB
AABB::operator+(const AABB &other) const noexcept {
    auto out = AABB{Eigen::Vector3f{thrust::min(min.x(), other.min.x()),
                                    thrust::min(min.y(), other.min.y()),
                                    thrust::min(min.z(), other.min.z())},
                    Eigen::Vector3f{thrust::max(max.x(), other.max.x()),
                                    thrust::max(max.y(), other.max.y()),
                                    thrust::max(max.z(), other.max.z())}};
#ifndef NDEBUG
    if (out.isFaulty()) {
        printf(
                "AABB IN THIS (%f %f %f %f %f %f) AND OTHER (%f %f %f %f %f "
                "%f) -> RESULT (%f %f "
                "%f %f %f %f)\n",
                min.x(), min.y(), min.z(), max.x(), max.y(), max.z(), other.min.x(),
                other.min.y(), other.min.z(), other.max.x(), other.max.y(),
                other.max.z(), min.x(), min.y(), min.z(), max.x(), max.y(),
                max.z());
    }
    assert(!out.isFaulty());
#endif
    return out;
}
__host__ __device__ bool AABB::isFaulty() const noexcept {
    constexpr float EMPTY_EPSILON = 1e-6f;
    const auto diff = max - min;

    // If any component is less than 0, the AABB is faulty
    if(diff[0] < 0 || diff[1] < 0 || diff[2] < 0) return true;

    // If all components are less than epsilon, the AABB is faulty
//    if(diff[0] <= EMPTY_EPSILON && diff[1] <= EMPTY_EPSILON && diff[2] <= EMPTY_EPSILON) return true;

    return false;
}