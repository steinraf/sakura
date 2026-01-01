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

    CPU_GPU BSDFQueryRecord() = default;
    CPU_GPU BSDFQueryRecord(Vec3f v) : wIn(v), wOut(v) {}
    CPU_GPU BSDFQueryRecord(Vec3f win, Vec3f wout, EMeasure measure) : wIn(win), wOut(wout), measure(measure) {}

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

    [[nodiscard]] CPU_GPU Color eval(const BSDFQueryRecord &query) const noexcept;
    [[nodiscard]] CPU_GPU float pdf(const BSDFQueryRecord &query) const noexcept;

    [[nodiscard]] __device__ Color sample(BSDFQueryRecord &bsdfQueryRecord, const Vec2f &sample) const noexcept;

    [[nodiscard]] CPU_GPU Color evalTexture(const Vec2f &uv) const noexcept;

    [[nodiscard]] CPU_GPU bool hasZeroTexture() const noexcept;
    CPU_GPU void setUnitTexture() noexcept;

private:
    Material material;
    Texture texture;
};
