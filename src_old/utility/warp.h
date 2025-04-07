//
// Created by steinraf on 21/10/22.
//


#pragma once

#include "../../src/common.h"
#include "sampler.h"
#include "vector.cuh"
#include <curand_kernel.h>

#define WARP_EPSILON 1e-10f

namespace Warp {

    [[nodiscard]] GPU_ONLY inline Vector3f sampleUniformHemisphere(Sampler &sampler, const Vector3f &pole) noexcept {
        // Naive implementation using rejection sampling
        Vector3f v;
        do {
            v[0] = 1.f - 2.f * sampler.getSample1D();
            v[1] = 1.f - 2.f * sampler.getSample1D();
            v[2] = 1.f - 2.f * sampler.getSample1D();
        } while(v.squaredNorm() > 1.f);

        if(v.dot(pole) < 0.f)
            v = -v;
        v /= v.norm();

        return v;
    }

    [[nodiscard]] CPU_GPU constexpr Vector2f squareToUniformSquare(const Vector2f &sample) noexcept {
        return sample;
    }

    [[nodiscard]] CPU_GPU constexpr float squareToUniformSquarePdf(const Vector2f &sample) noexcept {
        return 1.f;
    }

    [[nodiscard]] CPU_GPU constexpr Vector2f squareToUniformDisk(const Vector2f &sample) noexcept {
        const float r = sqrt(sample[0]);
        const float phi = (2 * sample[1] - 1) * M_PIf;
        return {r * sin(phi), r * cos(phi)};
    }

    [[nodiscard]] CPU_GPU constexpr float squareToUniformDiskPdf(const Vector2f &p) noexcept {
        return static_cast<float>(p.squaredNorm() <= 1) * M_1_PIf;
    }

    [[nodiscard]] CPU_GPU constexpr Vector3f squareToUniformSphereCap(const Vector2f &sample, float cosThetaMax) noexcept {
        const float cosT = sample[0] * (1 - cosThetaMax) + cosThetaMax;
        const float phi = 2 * M_PIf * sample[1];
        const float sTheta = sin(acos(cosT));
        return {
                sTheta * cos(phi),
                sTheta * sin(phi),
                cosT};
    }

    [[nodiscard]] CPU_GPU constexpr float squareToUniformSphereCapPdf(const Vector3f &v, float cosThetaMax) noexcept {
        return static_cast<float>(v[2] >= cosThetaMax) * M_1_PIf / (2.f - 2.f * cosThetaMax);
    }

    [[nodiscard]] CPU_GPU constexpr Vector3f squareToUniformSphere(const Vector2f &sample) noexcept{
        float cosT = 2 * sample[0] - 1;
        float phi = 2 * M_PIf * sample[1];
        float sTheta = sin(acos(cosT));
        return {
                sTheta * sin(phi),
                sTheta * cos(phi),
                cosT};
    }

    [[nodiscard]] CPU_GPU constexpr float squareToUniformSpherePdf(const Vector3f &v) noexcept {
        return static_cast<float>((v.squaredNorm() - 1.f) < EPSILON) * M_1_PIf / 4.f;
    }

    [[nodiscard]] CPU_GPU constexpr Vector3f squareToUniformHemisphere(const Vector2f &sample) noexcept {
        return squareToUniformSphereCap(sample, 0.f);
    }

    [[nodiscard]] CPU_GPU constexpr float squareToUniformHemispherePdf(const Vector3f &v) noexcept{
        return squareToUniformSphereCapPdf(v, 0);
    }

    [[nodiscard]] CPU_GPU constexpr Vector3f squareToCosineHemisphere(const Vector2f &sample) noexcept {
        auto p = squareToUniformDisk(sample);
        float z = sqrt(1 - p[0] * p[0] - p[1] * p[1]);
        return {p[0], p[1], z};
    }

    [[nodiscard]] CPU_GPU constexpr float squareToCosineHemispherePdf(const Vector3f &v) noexcept {
        return v[2] < 0 ? 0.f : v[2] * M_1_PIf;
    }

    [[nodiscard]] CPU_GPU constexpr Vector3f squareToBeckmann(const Vector2f &sample, float alpha) noexcept {
        const float cosTheta = sqrt(1.f / (1 - alpha * alpha * log(1 - sample[0])));
        const float phi = sample[1] * 2 * M_PIf;
        const float sTheta = sin(acos(cosTheta));
        return {
                sTheta * sin(phi),
                sTheta * cos(phi),
                cosTheta};
    }

    [[nodiscard]] CPU_GPU constexpr float squareToBeckmannPdf(const Vector3f &m, float alpha) noexcept {
        const float cosT3 = m[2] * m[2] * m[2];
        if(cosT3 < WARP_EPSILON) return 0;
        const float alphaI2 = 1.f / (alpha * alpha);
        return exp((1.f - 1.f / (m[2] * m[2])) * alphaI2) * M_1_PIf * alphaI2 / (cosT3);
    }

    [[nodiscard]] CPU_GPU constexpr Vector3f squareToUniformTriangle(const Vector2f &sample) noexcept {
        float su1 = sqrtf(sample[0]);
        float u = 1.f - su1, v = sample[1] * su1;
        return {u, v, 1.f - u - v};
    }

    [[nodiscard]] CPU_GPU constexpr size_t sampleCDF(float sample, const float *start, const float *ending) noexcept {

        const float *begin = start, *end = ending;

        size_t count = end - begin, step = 0;

        const float *it = nullptr;
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


        while(begin >= start && begin < ending){
            if((*(begin + 1) - *begin) == 0){
#ifndef NDEBUG
//                const int diff = begin - start;
                //TODO find a better fix for this
//                printf("DEBUG: PDF difference is zero for idx %i with value %f\n", diff, *begin);
#endif
                ++begin;

            }else{
                break;
            }
        }
        assert(begin <= ending);
        assert(begin >= start);

        const size_t idx = begin - start;




        return idx;
    }
}// namespace Warp