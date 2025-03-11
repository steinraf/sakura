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

    __host__ __device__ explicit Material() : type(MaterialType::DIFFUSE) {
    }

    __host__ __device__ explicit Material(MaterialType type) : type(type) {
    }

    __host__ __device__ explicit Material(MaterialType type, float iorInterior, float iorExterior) : type(type), ior{iorInterior, iorExterior} {
        assert(type == MaterialType::DIELECTRIC);
    }

    [[nodiscard]] __host__ __device__ static Material Diffuse() {
        return Material(MaterialType::DIFFUSE);
    }

    [[nodiscard]] __host__ __device__ Color eval(const Texture &texture, const BSDFQueryRecord &bsdfQueryRecord) const noexcept;
    [[nodiscard]] __host__ __device__ float pdf(const BSDFQueryRecord &bsdfQueryRecord) const noexcept;

    [[nodiscard]] __host__ __device__ float iorInterior() const noexcept {
        assert(type == MaterialType::DIELECTRIC);
        return ior.interior;
    }
    [[nodiscard]] __host__ __device__ float iorExterior() const noexcept {
        assert(type == MaterialType::DIELECTRIC);
        return ior.exterior;
    }

private:
    union {
        struct {
            float interior;
            float exterior;
        } ior;// index of refraction
    };
};
