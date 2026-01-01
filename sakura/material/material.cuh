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

    CPU_GPU explicit Material() : type(MaterialType::DIFFUSE) {
    }

    CPU_GPU explicit Material(MaterialType type) : type(type) {
    }

    CPU_GPU Material(const Material &other);
    CPU_GPU Material &operator=(const Material &other);

    CPU_GPU explicit Material(MaterialType type, float iorInterior, float iorExterior) : type(type), ior{iorInterior, iorExterior} {
        assert(type == MaterialType::DIELECTRIC);
    }

    CPU_GPU explicit Material(MaterialType type, float alpha, float iorInterior, float iorExterior, const Color &kd) : type(type), microfacet{alpha, iorInterior, iorExterior, kd, 1.0f - kd.maxCoeff()} {
        //Microfacet model taken from CG lecture
        assert(type == MaterialType::MICROFACET);
    }


    [[nodiscard]] CPU_GPU static Material Diffuse() {
        return Material(MaterialType::DIFFUSE);
    }

    [[nodiscard]] CPU_GPU Color eval(const Texture &texture, const BSDFQueryRecord &bsdfQueryRecord) const noexcept;
    [[nodiscard]] CPU_GPU float pdf(const BSDFQueryRecord &bsdfQueryRecord) const noexcept;

    [[nodiscard]] CPU_GPU float iorInterior() const noexcept {
        assert(type == MaterialType::DIELECTRIC);
        return ior.interior;
    }
    [[nodiscard]] CPU_GPU float iorExterior() const noexcept {
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

    [[nodiscard]] CPU_GPU float evalBeckmann(const Vec3f &n) const;
    [[nodiscard]] CPU_GPU float smithBeckmannG1(const Vec3f &v, const Vec3f &n) const;
    [[nodiscard]] CPU_GPU static float fresnel(float cosTheta, float extIOR, float intIOR);
};
