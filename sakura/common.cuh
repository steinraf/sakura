//
// Created by steinraf on 17.02.25.
//

#pragma once

#include <Eigen/Dense>
#include <cuda/std/atomic>

// Forward declarations

struct AABB;
class AreaLight;
class BSDF;
class BLAS;
struct BSDFQueryRecord;
class BVH;
class Camera;
struct EmitterQueryRecord;
struct FeatureBuffer;
struct Film;
class Frame;
struct Intersection;
class Material;
class Normal3f;
struct Ray;
class Renderable;
class Sampler;
struct Sensor;
class Scene;
class SceneBuilder;
struct ShapeQueryRecord;
class Texture;
class TLAS;
class Triangle;

using Vec2f = Eigen::Vector2f;
using Vec3f = Eigen::Vector3f;
using Color = Vec3f;

#define CPU_GPU __host__ __device__

void checkCudaErrors(cudaError result);

CPU_GPU int safe_uint_to_int(unsigned int i);

template<typename T>
concept StatType = std::is_floating_point_v<T> || std::is_same_v<T, Vec3f>;

template<StatType T>
class Statistic {
public:
    CPU_GPU Statistic() : numElements(0), mean(getZero()), variance(getZero()) {}
    CPU_GPU Statistic(const Statistic<T> &other) = default;
    CPU_GPU Statistic(Statistic<T> &&other) = default;
    CPU_GPU Statistic &operator=(const Statistic<T> &other) = default;
    CPU_GPU ~Statistic() = default;

    // Welford's online algorithm to incrementally calculate variance
    CPU_GPU void addElement(T element) {
        numElements++;
        T delta = element - mean;
        mean += delta / numElements;
        if constexpr(std::is_same_v<T, Vec3f>) {
            variance += delta.cwiseProduct(element - mean);
        } else {
            variance += delta * (element - mean);
        }
    }

    [[nodiscard]] CPU_GPU T getMean() const {
        return mean;
    }

    [[nodiscard]] CPU_GPU T getVariance() const {
        if(numElements == 0) {
            return getZero();
        }
        return variance / numElements;
    }

    [[nodiscard]] CPU_GPU T getSampleVariance() const {
        if(numElements <= 1) {
            return getZero();
        }
        return variance / (numElements - 1);
    }

    [[nodiscard]] CPU_GPU size_t getNumElements() const {
        return numElements;
    }

    CPU_GPU void clear() {
        numElements = 0;
        mean = getZero();
        variance = getZero();
    }

    CPU_GPU void scale(float factor) {
        numElements *= factor;
    }

private:
    CPU_GPU T getZero() const {
        if constexpr(std::is_same_v<T, Vec3f>) {
            return Vec3f::Zero();
        } else {
            return T(0);
        }
    }

    size_t numElements;
    T mean;
    T variance;
};
