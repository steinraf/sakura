//
// Created by steinraf on 25.02.25.
//

#pragma once

#include "../common.cuh"

enum class MaterialType {
    DIFFUSE,
    SPECULAR,
    DIELECTRIC,
};

class Material {
public:
    MaterialType type;

    [[nodiscard]] __host__ __device__ Color eval(const Texture &texture, const BSDFQueryRecord &bsdfQueryRecord) const noexcept;
    [[nodiscard]] __host__ __device__ float pdf(const BSDFQueryRecord &bsdfQueryRecord) const noexcept;
    
private:
    union {
        struct {
            float interior;
            float exterior;
        } ior;// index of refraction
    };
};
