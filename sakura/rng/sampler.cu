//
// Created by steinraf on 05.02.25.
//

#include "sampler.cuh"

__device__ Sampler::Sampler(curandState *state) : rng(state) {}

[[nodiscard]] __device__ float Sampler::getSample1D() noexcept {
    return curand_uniform(rng);
}

[[nodiscard]] __device__ Eigen::Vector2f Sampler::getSample2D() noexcept {
    return Eigen::Vector2f{curand_uniform(rng), curand_uniform(rng)};
}

[[nodiscard]] __device__ Eigen::Vector3f Sampler::getSample3D() noexcept {
    return Eigen::Vector3f{curand_uniform(rng), curand_uniform(rng),
                           curand_uniform(rng)};
}

namespace sample {
    [[nodiscard]] __device__ Eigen::Vector3f uniformHemisphere(
            Sampler &sampler, const Eigen::Vector3f &pole) noexcept {
        // Naive implementation using rejection sampling
        // TODO change
        Eigen::Vector3f v;
        do {
            v[0] = 1.f - 2.f * sampler.getSample1D();
            v[1] = 1.f - 2.f * sampler.getSample1D();
            v[2] = 1.f - 2.f * sampler.getSample1D();
        } while (v.squaredNorm() > 1.f);

        if (v.dot(pole) < 0.f) v = -v;
        v /= v.norm();

        return v;
    }

    [[nodiscard]] __device__ float squareToUniformSphereCapPdf(
            const Eigen::Vector3f &v, float cosThetaMax) noexcept {
        return static_cast<float>(v[2] >= cosThetaMax) * M_1_PIf /
               (2.f - 2.f * cosThetaMax);
    }

    [[nodiscard]] __device__ Eigen::Vector3f squareToUniformSphereCap(
            const Eigen::Vector2f &sample, float cosThetaMax) noexcept {
        const float cosT = sample[0] * (1 - cosThetaMax) + cosThetaMax;
        const float phi = 2 * M_PIf * sample[1];
        const float sTheta = sin(acos(cosT));
        return {sTheta * cos(phi), sTheta * sin(phi), cosT};
    }

    [[nodiscard]] __device__ Eigen::Vector2f squareToUniformDisk(
            const Eigen::Vector2f &sample) noexcept {
        const float r = sqrt(sample[0]);
        const float phi = (2 * sample[1] - 1) * M_PIf;
        return {r * sin(phi), r * cos(phi)};
    }

    [[nodiscard]] __device__ Eigen::Vector3f squareToCosineHemisphere(
            const Eigen::Vector2f &sample) noexcept {
        auto p = squareToUniformDisk(sample);
        float z = sqrt(1 - p[0] * p[0] - p[1] * p[1]);
        return {p[0], p[1], z};
    }

    [[nodiscard]] __device__ float squareToCosineHemispherePdf(
            const Eigen::Vector3f &v) noexcept {
        return v[2] < 0 ? 0.f : v[2] * M_1_PIf;
    }

}  // namespace sample
