//
// Created by steinraf on 26.02.25.
//

#pragma once

#include "../common.cuh"
#include "../texture/texture.cuh"
#include "material.cuh"

enum class EMeasure {
    EUnknownMeasure,
    ESolidAngle,
    EDiscrete
};

struct BSDFQueryRecord {
public:
    Vec3f wIn; // incident light omega
    Vec3f wOut;// outgoing light omega

    Vec3f position;// query position
    Vec2f uv;      // query uv

    float eta = 1.0;// index of refraction

    EMeasure measure = EMeasure::EUnknownMeasure;

    __host__ __device__ BSDFQueryRecord() = default;
    __host__ __device__ BSDFQueryRecord(Vec3f v) : wIn(v), wOut(v) {}

private:
};

class BSDF {
public:
    explicit BSDF(Material material, Texture texture);
    BSDF() = default;
    BSDF(const BSDF &other) = default;
    BSDF(BSDF &&other) = default;
    BSDF &operator=(const BSDF &other) = default;
    BSDF &operator=(BSDF &&other) = default;
    ~BSDF() = default;

    [[nodiscard]] __host__ __device__ Color eval(const BSDFQueryRecord &query) const noexcept;
    [[nodiscard]] __host__ __device__ float pdf(const BSDFQueryRecord &query) const noexcept;

    [[nodiscard]] __device__ Color sample(BSDFQueryRecord &bsdfQueryRecord, const Vec2f &sample) const noexcept;

private:
    Material material;
    Texture texture;
};
