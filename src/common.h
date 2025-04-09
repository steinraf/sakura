//
// Created by steinraf on 07.04.25.
//



#pragma once

#include "../src_old/utility/vector.cuh"

class AreaLight;
class BLAS;
struct EmitterQueryRecord;
class Ray3f;
class Scene;
struct ShapeQueryRecord;

using Vec2f = Vector2f;
using Vec3f = Vector3f;




__host__ inline void check_cuda(cudaError_t result, char const *const func, const char *const file, int line) noexcept(false) {
    if(result) {
        std::cerr << "CUDA error = " << static_cast<unsigned int>(result) << ":" << cudaGetErrorString(result) << "\nAt " << file << ":" << line << " '" << func << "' \n";
        cudaDeviceReset();
        exit(99);
    }
}

#define checkCudaErrors(val) check_cuda((val), #val, __FILE__, __LINE__)


template<typename T>
concept StatType = std::is_floating_point_v<T> || std::is_same_v<T, Vec3f>;

template<StatType T>
class Statistic {
public:
    CPU_GPU Statistic() : numElements(0), mean(getZero()), variance(getZero()) {}
    Statistic(const Statistic<T> &other) = default;
    Statistic(Statistic<T> &&other) = default;
    Statistic &operator=(const Statistic<T> &other) = default;
    ~Statistic() = default;

    // Welford's online algorithm to incrementally calculate variance
    CPU_GPU void addElement(T element) {
        if constexpr (std::is_same_v<T, Vec3f>) {
            if(!isfinite(element[0]) || !isfinite(element[1]) || !isfinite(element[2])) {
                return;
            }
        } else {
            if(!isfinite(element)) {
                return;
            }
        }
        numElements++;
        T delta = element - mean;
        mean += delta / numElements;
        variance += delta * (element - mean);
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

    CPU_GPU void manipulate(float scale) const {
        mean *= scale;
        variance *= scale * scale;
    }

    CPU_GPU void manipulateMean(T newMean) {
        T delta = newMean - mean;
        mean = newMean;
        variance += delta * delta * numElements;
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

