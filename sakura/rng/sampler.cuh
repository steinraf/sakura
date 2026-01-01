//
// Created by steinraf on 05.02.25.
//

#pragma once

#include <curand_kernel.h>

#include <Eigen/Dense>

#include "../common.cuh"

// Sampler class wrapping curandState
class Sampler {
public:
    __device__ explicit Sampler(curandState *state);

    [[nodiscard]] __device__ float getSample1D() noexcept;

    [[nodiscard]] __device__ Vec2f getSample2D() noexcept;

    [[nodiscard]] __device__ Vec3f getSample3D() noexcept;

private:
    curandState *rng;
};

namespace sample {

    // Naive Sampling of a uniform hemisphere
    [[nodiscard]] __device__ Vec3f uniformHemisphere(
            Sampler &sampler, const Vec3f &pole) noexcept;

    [[nodiscard]] CPU_GPU Vec3f squareToUniformSphere(const Vec2f &sample) noexcept;

    [[nodiscard]] CPU_GPU float squareToUniformSphereCapPdf(
            const Vec3f &v, float cosThetaMax) noexcept;

    [[nodiscard]] CPU_GPU Vec3f squareToUniformSphereCap(
            const Vec2f &sample, float cosThetaMax) noexcept;

    [[nodiscard]] CPU_GPU Vec2f squareToUniformDisk(
            const Vec2f &sample) noexcept;

    [[nodiscard]] CPU_GPU Vec3f squareToUniformTriangle(
            const Vec2f &sample) noexcept;

    [[nodiscard]] CPU_GPU Vec3f squareToCosineHemisphere(
            const Vec2f &sample) noexcept;
    [[nodiscard]] CPU_GPU float squareToCosineHemispherePdf(
            const Vec3f &v) noexcept;

    [[nodiscard]] CPU_GPU size_t sampleCDF(float sample, float *cdf, size_t cdfSize) noexcept;

}// namespace sample
