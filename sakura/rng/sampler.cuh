//
// Created by steinraf on 05.02.25.
//

#pragma once

#include <curand_kernel.h>

#include <Eigen/Dense>

// Sampler class wrapping curandState
class Sampler {
public:
    __device__ explicit Sampler(curandState *state);

    [[nodiscard]] __device__ float getSample1D() noexcept;

    [[nodiscard]] __device__ Eigen::Vector2f getSample2D() noexcept;

    [[nodiscard]] __device__ Eigen::Vector3f getSample3D() noexcept;

private:
    curandState *rng;
};

namespace sample {

    // Naive Sampling of a uniform hemisphere
    [[nodiscard]] __device__ Eigen::Vector3f uniformHemisphere(
            Sampler &sampler, const Eigen::Vector3f &pole) noexcept;

    [[nodiscard]] __device__ float squareToUniformSphereCapPdf(
            const Eigen::Vector3f &v, float cosThetaMax) noexcept;

    [[nodiscard]] __device__ Eigen::Vector3f squareToUniformSphereCap(
            const Eigen::Vector2f &sample, float cosThetaMax) noexcept;

    [[nodiscard]] __device__ Eigen::Vector2f squareToUniformDisk(
            const Eigen::Vector2f &sample) noexcept;

    [[nodiscard]] __device__ Eigen::Vector3f squareToCosineHemisphere(
            const Eigen::Vector2f &sample) noexcept;
    [[nodiscard]] __device__ float squareToCosineHemispherePdf(
            const Eigen::Vector3f &v) noexcept;

}// namespace sample
