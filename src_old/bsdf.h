//
// Created by steinraf on 24/10/22.
//

#pragma once

#include "textures/imageTexture.h"
#include "utility/frame.h"
#include "utility/ray.h"
#include "utility/sampler.h"
#include "utility/warp.h"

//Nori Fresnel Term
[[nodiscard]] __device__ constexpr float fresnel(float cosThetaI, float extIOR, float intIOR) noexcept {
    float etaI = extIOR, etaT = intIOR;

    if(extIOR == intIOR)
        return 0.0f;

    /* Swap the indices of refraction if the interaction starts
   at the inside of the object */
    if(cosThetaI < 0.0f) {
        cuda::std::swap(etaI, etaT);
        cosThetaI = -cosThetaI;
    }

    /* Using Snell's law, calculate the squared sine of the
   angle between the normal and the transmitted ray */
    float eta = etaI / etaT,
          sinThetaTSqr = eta * eta * (1 - cosThetaI * cosThetaI);

    if(sinThetaTSqr > 1.0f)
        return 1.0f; /* Total internal reflection! */

    float cosThetaT = std::sqrt(1.0f - sinThetaTSqr);

    float Rs = (etaI * cosThetaI - etaT * cosThetaT) / (etaI * cosThetaI + etaT * cosThetaT);
    float Rp = (etaT * cosThetaI - etaI * cosThetaT) / (etaT * cosThetaI + etaI * cosThetaT);

    return (
                   Rs * Rs +
                   Rp * Rp) /
           2.0f;
}

enum EMeasure {
    EUnknownMeasure = 0,
    ESolidAngle,
    EDiscrete
};

struct BSDFQueryRecord {
    Vector3f wi;
    Vector3f wo;

    Vector3f p;
    Vector2f uv;

    float eta;

    EMeasure measure;

    __device__ constexpr explicit BSDFQueryRecord(const Vector3f &wi) noexcept
        : wi(wi), wo(0.f),
          uv(0.f), eta(1.0),
          measure(EUnknownMeasure) {}

    __device__ constexpr BSDFQueryRecord(const Vector3f &wi, const Vector3f &wo, EMeasure measure) noexcept
        : wi(wi), wo(wo), uv(0.f), eta(1.0), measure(measure) {
    }
};


enum class Material {
    DIFFUSE,
    MIRROR,
    DIELECTRIC
};


class BSDF {
public:
    Material material;
    Texture texture;
    float m_intIOR, m_extIOR;

    __device__ __host__ constexpr BSDF() noexcept
        : material(Material::DIFFUSE), texture(Vector3f{1.f}), m_intIOR(1.5046f), m_extIOR(1.000277f) {
    }

    __device__ __host__ constexpr BSDF(Material mat, Color3f albedo) noexcept
        : material(mat), texture(albedo), m_intIOR(1.5046f), m_extIOR(1.000277f) {
    }

    __host__ BSDF(Material mat, const std::filesystem::path &texturePath) noexcept
        : material(mat), texture(texturePath), m_intIOR(1.5046f), m_extIOR(1.000277f) {
    }

    //Overload to make nori-syntax possible
    __device__ __host__ const BSDF* operator->() const noexcept {
        return this;
    }
    __device__ __host__ BSDF* operator->() noexcept {
        return this;
    }

    [[nodiscard]] __device__ constexpr Color3f getAlbedo(const Vector2f &uv) const noexcept {
        return texture.eval(uv);
    }

    [[nodiscard]] __device__ constexpr Color3f eval(const BSDFQueryRecord &bsdfQueryRecord) const noexcept {
        switch(material) {
            case Material::DIFFUSE:
                if(bsdfQueryRecord.measure != ESolidAngle
                    || Frame::cosTheta(bsdfQueryRecord.wi) <= 0
                    || Frame::cosTheta(bsdfQueryRecord.wo) <= 0)
                    return Color3f(0.0f);

                return texture.eval(bsdfQueryRecord.uv) * M_1_PIf;
            case Material::MIRROR:
                return Color3f{0.f};
            case Material::DIELECTRIC:
                return Color3f{0.f};
        }
    }

    [[nodiscard]] __device__ constexpr float pdf(const BSDFQueryRecord &bsdfQueryRecord) const noexcept {
        switch(material) {
            case Material::DIFFUSE:
                if(bsdfQueryRecord.measure != ESolidAngle || Frame::cosTheta(bsdfQueryRecord.wi) <= 0 || Frame::cosTheta(bsdfQueryRecord.wo) <= 0)
                    return 0.0f;

                return M_1_PIf * Frame::cosTheta(bsdfQueryRecord.wo);
            case Material::MIRROR:
                return 0.f;
            case Material::DIELECTRIC:
                return 0.f;
        }
    }


    [[nodiscard]] __device__ constexpr Color3f sample(BSDFQueryRecord &bsdfQueryRecord, const Vector2f &sample) const noexcept {
        switch(material) {
            case Material::DIFFUSE:
                if(Frame::cosTheta(bsdfQueryRecord.wi) <= 0)
                    return Color3f(0.0f);

                bsdfQueryRecord.measure = ESolidAngle;

                bsdfQueryRecord.wo = Warp::squareToCosineHemisphere(sample);

                bsdfQueryRecord.eta = 1.0f;

                return texture.eval(bsdfQueryRecord.uv);
            case Material::MIRROR:
                if(Frame::cosTheta(bsdfQueryRecord.wi) <= 0)
                    return Color3f{0.0f};

                bsdfQueryRecord.wo = Vector3f{
                        -bsdfQueryRecord.wi[0],
                        -bsdfQueryRecord.wi[1],
                        bsdfQueryRecord.wi[2]};
                bsdfQueryRecord.measure = EDiscrete;

                bsdfQueryRecord.eta = 1.0f;

                return Color3f{1.0f};
            case Material::DIELECTRIC:
                float extIOR = m_extIOR, intIOR = m_intIOR, cosThetaI = Frame::cosTheta(bsdfQueryRecord.wi);
                Vector3f normal{0.f, 0.f, 1.f};
                if(Frame::cosTheta(bsdfQueryRecord.wi) < 0) {
                    extIOR = m_intIOR;
                    intIOR = m_extIOR;
                    cosThetaI *= -1;
                    normal *= -1;
                }


                const float fresnelCoeff = fresnel(cosThetaI, extIOR, intIOR);

                bsdfQueryRecord.measure = EDiscrete;

                if(sample[0] < fresnelCoeff) {
                    bsdfQueryRecord.eta = 1.f;

                    bsdfQueryRecord.wo = Vector3f(
                            -bsdfQueryRecord.wi[0],
                            -bsdfQueryRecord.wi[1],
                            bsdfQueryRecord.wi[2]);

                    return Color3f{1.f};

                } else {
                    bsdfQueryRecord.eta = extIOR / intIOR;

                    bsdfQueryRecord.wo =
                            -bsdfQueryRecord.eta * (bsdfQueryRecord.wi - (bsdfQueryRecord.wi.dot(normal) * normal))
                            - normal * sqrt(1 - bsdfQueryRecord.eta * bsdfQueryRecord.eta * (1 - bsdfQueryRecord.wi[2] * bsdfQueryRecord.wi[2]));

                    return Color3f{bsdfQueryRecord.eta * bsdfQueryRecord.eta};
                }
        }
    }

    [[nodiscard]] __host__ __device__ constexpr bool isDeltaDistribution() const noexcept{
            switch(material){
            break;case Material::DIFFUSE:
                return false;
            break;case Material::MIRROR:
                return true;
            break;case Material::DIELECTRIC:
                return true;
            }
    };
};
