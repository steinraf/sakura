//
// Created by steinraf on 07.04.25.
//

#pragma once

#include "../common.h"
#include "../statistics/statistics.h"

enum class BUFFERTYPE {
    MEAN,
    VARIANCE,
    SAMPLEVARIANCE,
    NUM_ELEMENTS
};




struct FeatureBuffer {
    FeatureBuffer() = delete;
    explicit __host__ FeatureBuffer(size_t numElements);
    __host__ ~FeatureBuffer();

    void __host__ clear();
    void __host__ decay(float k);

    size_t numElements;
    Statistic<Vec3f> *color;
    Statistic<Vec3f> *normal;
    Statistic<Vec3f> *position;
    Statistic<Vec3f> *albedo;
    Statistic<Vec3f> *uv;
};

__global__ void clearFeatureBuffer(FeatureBuffer *buffer);

template<BUFFERTYPE B>
__global__ void extractBufferInfo(const Statistic<Vec3f> *stat, Vec3f *out, int width, int height, auto f) {

    size_t pixelIndex = blockIdx.x * blockDim.x + threadIdx.x;

    if(pixelIndex >= width * height) return;

    size_t x = width - 1 - pixelIndex % width, y = pixelIndex / width;


    auto color = [&]() -> Vec3f {
        if constexpr(B == BUFFERTYPE::MEAN) return f(stat[pixelIndex].getMean());
        if constexpr(B == BUFFERTYPE::VARIANCE) return f(stat[pixelIndex].getVariance());
        if constexpr(B == BUFFERTYPE::SAMPLEVARIANCE) return f(stat[pixelIndex].getSampleVariance());
        if constexpr(B == BUFFERTYPE::NUM_ELEMENTS) return f(Vec3f{float(stat[pixelIndex].getNumElements()), float(stat[pixelIndex].getNumElements()), float(stat[pixelIndex].getNumElements())});
        else
            return Vec3f{-1, -1, -1};
    }();

    out[pixelIndex] = color;
}