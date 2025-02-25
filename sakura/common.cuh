//
// Created by steinraf on 17.02.25.
//

#pragma once

#include <Eigen/Dense>

// Forward declarations

struct AABB;
class BVH;
class Camera;
struct FeatureBuffer;
struct Film;
class Frame;
struct Intersection;
class Ray;
class Renderable;
class Sampler;
struct Sensor;
class Scene;
class SceneBuilder;
class Triangle;

using Vec3f = Eigen::Vector3f;


void checkCudaErrors(cudaError result);

__host__ __device__ int safe_uint_to_int(unsigned int i);

template<typename T>
concept StatType = std::is_floating_point_v<T> || std::is_same_v<T, Vec3f>;

template<StatType T>
class Statistic {
public:
    __host__ __device__ Statistic() : numElements(0), mean(getZero()), variance(getZero()) {}
    __host__ __device__ Statistic(const Statistic<T> &other) = default;
    __host__ __device__ Statistic(Statistic<T> &&other) = default;
    __host__ __device__ Statistic &operator=(const Statistic<T> &other) = default;
    __host__ __device__ ~Statistic() = default;

    // Welford's online algorithm to incrementally calculate variance
    __host__ __device__ void addElement(T element) {
        numElements++;
        T delta = element - mean;
        mean += delta / numElements;
        if constexpr(std::is_same_v<T, Vec3f>) {
            variance += delta.cwiseProduct(element - mean);
        } else {
            variance += delta * (element - mean);
        }
    }

    [[nodiscard]] __host__ __device__ T getMean() const {
        return mean;
    }

    [[nodiscard]] __host__ __device__ T getVariance() const {
        if(numElements == 0) {
            return getZero();
        }
        return variance / numElements;
    }

    [[nodiscard]] __host__ __device__ T getSampleVariance() const {
        if(numElements <= 1) {
            return getZero();
        }
        return variance / (numElements - 1);
    }

    [[nodiscard]] __host__ __device__ size_t getNumElements() const {
        return numElements;
    }

    __host__ __device__ void clear() {
        numElements = 0;
        mean = getZero();
        variance = getZero();
    }

private:
    __host__ __device__ T getZero() const {
        if constexpr(std::is_same_v<T, Vec3f>) {
            return Vec3f::Zero();
        } else {
            return T();
        }
    }

    size_t numElements;
    T mean;
    T variance;
};