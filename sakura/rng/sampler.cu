//
// Created by steinraf on 05.02.25.
//

#include "sampler.cuh"

__device__ Sampler::Sampler(curandState *state) : rng(state) {}

[[nodiscard]] __device__ float Sampler::getSample1D() noexcept {
    return curand_uniform(rng);
}

[[nodiscard]] __device__ Vec2f Sampler::getSample2D() noexcept {
    return Vec2f{curand_uniform(rng), curand_uniform(rng)};
}

[[nodiscard]] __device__ Vec3f Sampler::getSample3D() noexcept {
    return Vec3f{curand_uniform(rng), curand_uniform(rng),
                 curand_uniform(rng)};
}

namespace sample {
    [[nodiscard]] __device__ Vec3f uniformHemisphere(
            Sampler &sampler, const Vec3f &pole) noexcept {
        // Naive implementation using rejection sampling
        // TODO change
        Vec3f v;
        do {
            v[0] = 1.f - 2.f * sampler.getSample1D();
            v[1] = 1.f - 2.f * sampler.getSample1D();
            v[2] = 1.f - 2.f * sampler.getSample1D();
        } while(v.squaredNorm() > 1.f);

        if(v.dot(pole) < 0.f) v = -v;
        v /= v.norm();

        return v;
    }

    [[nodiscard]] CPU_GPU Vec3f squareToUniformSphere(const Vec2f &sample) noexcept {
        float cosT = 2.f * sample[0] - 1.f;
        float phi = 2.f * M_PIf * sample[1];
        float sinT = sin(acos(cosT));
        return {sinT * sin(phi), sinT * cos(phi), cosT};
    }


    [[nodiscard]] CPU_GPU float squareToUniformSphereCapPdf(
            const Vec3f &v, float cosThetaMax) noexcept {
        return static_cast<float>(v[2] >= cosThetaMax) * M_1_PIf /
               (2.f - 2.f * cosThetaMax);
    }

    [[nodiscard]] CPU_GPU Vec3f squareToUniformSphereCap(
            const Vec2f &sample, float cosThetaMax) noexcept {
        const float cosT = sample[0] * (1 - cosThetaMax) + cosThetaMax;
        const float phi = 2 * M_PIf * sample[1];
        const float sTheta = sin(acos(cosT));
        return {sTheta * cos(phi), sTheta * sin(phi), cosT};
    }

    [[nodiscard]] CPU_GPU Vec2f squareToUniformDisk(
            const Vec2f &sample) noexcept {
        const float r = sqrt(sample[0]);
        const float phi = (2 * sample[1] - 1) * M_PIf;
        return {r * sin(phi), r * cos(phi)};
    }

    [[nodiscard]] CPU_GPU Vec3f squareToUniformTriangle(
            const Vec2f &sample) noexcept {
        float s = sqrtf(sample[0]);
        float u = 1 - s;
        float v = sample[1] * s;
        return {u, v, 1 - u - v};
    }

    [[nodiscard]] CPU_GPU Vec3f squareToCosineHemisphere(
            const Vec2f &sample) noexcept {
        auto p = squareToUniformDisk(sample);
        float z = sqrt(1 - p[0] * p[0] - p[1] * p[1]);
        return {p[0], p[1], z};
    }

    [[nodiscard]] CPU_GPU float squareToCosineHemispherePdf(
            const Vec3f &v) noexcept {
        return v[2] < 0 ? 0.f : v[2] * M_1_PIf;
    }

    [[nodiscard]] CPU_GPU size_t sampleCDF(float sample, float *cdf, size_t cdfSize) noexcept {
        const float *begin = cdf;
        size_t count = cdfSize - 1, step = 0;

        const float *it;
        while(count > 0) {
            it = begin;
            step = count / 2;
            it += step;
            if(*it < sample) {
                begin = ++it;
                count -= step + 1;
            } else {
                count = step;
            }
        }

#ifndef NDEBUG
        int counter = 0;
#endif

        // We dont want to return a cdf with zero delta
        while(begin >= cdf && begin < cdf + cdfSize - 1) {
            if((*(begin + 1) - *begin) == 0.0f) {
                ++begin;
#ifndef NDEBUG
                ++counter;
#endif
            } else
                break;
        }

#ifndef NDEBUG
        if(counter > 0) {
            printf("Skipped %d elements\n", counter);
        }
#endif

        assert(begin >= cdf);
        assert(begin < cdf + cdfSize);


        return begin - cdf;
    }


}// namespace sample
