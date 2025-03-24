//
// Created by steinraf on 25.02.25.
//

#pragma once

#include "../common.cuh"

enum class MaterialType {
    DIFFUSE,
    SPECULAR,
    MICROFACET,
    DIELECTRIC,
};

class Material {
public:
    MaterialType type;

    __host__ __device__ explicit Material() : type(MaterialType::DIFFUSE) {
    }

    __host__ __device__ explicit Material(MaterialType type) : type(type) {
    }

    __host__ __device__ Material(const Material &other);
    __host__ __device__ Material &operator=(const Material &other);

    __host__ __device__ explicit Material(MaterialType type, float iorInterior, float iorExterior) : type(type), ior{iorInterior, iorExterior} {
        assert(type == MaterialType::DIELECTRIC);
    }

    __host__ __device__ explicit Material(MaterialType type, float alpha, float iorInterior, float iorExterior, const Color &kd) : type(type), microfacet{alpha, iorInterior, iorExterior, kd, 1.0f - kd.maxCoeff()} {
        //Microfacet model taken from CG lecture
        assert(type == MaterialType::MICROFACET);
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
        struct {
            float alpha;
            float interior;
            float exterior;
            Color kd;
            float ks;
        } microfacet;
    };

    [[nodiscard]] __host__ __device__ float evalBeckmann(const Vec3f &n) const;
    [[nodiscard]] __host__ __device__ float smithBeckmannG1(const Vec3f &v, const Vec3f &n) const;
    [[nodiscard]] __host__ __device__ static float fresnel(float cosTheta, float extIOR, float intIOR);
};
