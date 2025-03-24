//
// Created by steinraf on 25.02.25.
//

#include "material.cuh"

#include "../geometry/intersection.cuh"
#include "../texture/texture.cuh"
#include "bsdf.cuh"

__host__ __device__ Color Material::eval(const Texture &texture, const BSDFQueryRecord &bsdfQueryRecord) const noexcept {

    Vec3f halfway = (bsdfQueryRecord.wIn + bsdfQueryRecord.wOut).normalized();

    switch(type) {
        case MaterialType::DIFFUSE:
            if(bsdfQueryRecord.measure != EMeasure::EDiscrete || Frame::cosTheta(bsdfQueryRecord.wIn) <= 0 || Frame::cosTheta(bsdfQueryRecord.wOut) <= 0)
                return {0.0f, 0.0f, 0.0f};

            return texture.eval(bsdfQueryRecord.uv) * M_1_PIf;
        case MaterialType::SPECULAR:
        case MaterialType::DIELECTRIC:
            return {0.0f, 0.0f, 0.0f};
        case MaterialType::MICROFACET:
            return microfacet.kd * M_1_PIf + Vec3f::Constant(microfacet.ks * evalBeckmann(halfway) * fresnel(halfway.dot(bsdfQueryRecord.wIn), microfacet.exterior, microfacet.interior) * smithBeckmannG1(bsdfQueryRecord.wIn, halfway) * smithBeckmannG1(bsdfQueryRecord.wOut, halfway) / (4.f * Frame::cosTheta(bsdfQueryRecord.wIn) * Frame::cosTheta(bsdfQueryRecord.wOut)));
        default:
            assert(false);
            return {0.0f, 0.0f, 0.0f};
    }
}
__host__ __device__ float Material::pdf(const BSDFQueryRecord &bsdfQueryRecord) const noexcept {
    const Vec3f halfway = (bsdfQueryRecord.wIn + bsdfQueryRecord.wOut).normalized();
    switch(type) {
        case MaterialType::DIFFUSE:
            if(bsdfQueryRecord.measure != EMeasure::EDiscrete || Frame::cosTheta(bsdfQueryRecord.wIn) <= 0 || Frame::cosTheta(bsdfQueryRecord.wOut) <= 0)
                return 0.0f;

            return Frame::cosTheta(bsdfQueryRecord.wOut) * M_1_PIf;
        case MaterialType::SPECULAR:
        case MaterialType::DIELECTRIC:
            return 0.0f;

        case MaterialType::MICROFACET:
            if(Frame::cosTheta(bsdfQueryRecord.wOut) <= 0)
                return 0.0f;
            return microfacet.ks * evalBeckmann(halfway) * Frame::cosTheta(halfway) / (4.f * halfway.dot(bsdfQueryRecord.wOut)) + (1.f - microfacet.ks) * Frame::cosTheta(bsdfQueryRecord.wOut) * M_1_PIf;

        default:
            assert(false);
            return 0.0f;
    }
}
__host__ __device__ float Material::evalBeckmann(const Vec3f &n) const {
    assert(type == MaterialType::MICROFACET);
    float ct = Frame::cosTheta(n),
          ct2 = ct * ct,
          k = Frame::tanTheta(n) / microfacet.alpha;
    return expf(-k * k) / (M_PIf * microfacet.alpha * microfacet.alpha * ct2 * ct2);
}

__host__ __device__ float Material::smithBeckmannG1(const Vec3f &v, const Vec3f &n) const {
    assert(type == MaterialType::MICROFACET);
    float tanTheta = Frame::tanTheta(v);

    if(tanTheta == 0)
        return 1.0f;

    if(n.dot(v) * Frame::cosTheta(v) <= 0)
        return 0.0f;

    float a = 1.0f / (microfacet.alpha * tanTheta);
    if(a >= 1.6f)
        return 1.0f;
    float a2 = a * a;

    return (3.535f * a + 2.181f * a2) / (1.0f + 2.276f * a + 2.577f * a2);
}
__host__ __device__ float Material::fresnel(float cosTheta, float extIOR, float intIOR) {
    // Nori fresnel
    float etaI = extIOR, etaT = intIOR;

    if(extIOR == intIOR)
        return 0.0f;

    /* Swap the indices of refraction if the interaction starts
       at the inside of the object */
    if(cosTheta < 0.0f) {
        std::swap(etaI, etaT);
        cosTheta = -cosTheta;
    }

    /* Using Snell's law, calculate the squared sine of the
       angle between the normal and the transmitted ray */
    float eta = etaI / etaT,
          sinThetaTSqr = eta * eta * (1 - cosTheta * cosTheta);

    if(sinThetaTSqr > 1.0f)
        return 1.0f; /* Total internal reflection! */

    float cosThetaT = std::sqrt(1.0f - sinThetaTSqr);

    float Rs = (etaI * cosTheta - etaT * cosThetaT) / (etaI * cosTheta + etaT * cosThetaT);
    float Rp = (etaT * cosTheta - etaI * cosThetaT) / (etaT * cosTheta + etaI * cosThetaT);

    return (Rs * Rs + Rp * Rp) / 2.0f;
}
__host__ __device__ Material::Material(const Material &other) : type(other.type) {
    switch(type) {
        case MaterialType::DIFFUSE:
        case MaterialType::SPECULAR:
            break;
        case MaterialType::DIELECTRIC:
            ior = other.ior;
            break;
        case MaterialType::MICROFACET:
            microfacet = other.microfacet;
            break;
        default:
            assert(false);
    }
}
__host__ __device__ Material &Material::operator=(const Material &other) {
    type = other.type;
    switch(type) {
        case MaterialType::DIFFUSE:
        case MaterialType::SPECULAR:
            break;
        case MaterialType::DIELECTRIC:
            ior = other.ior;
            break;
        case MaterialType::MICROFACET:
            microfacet = other.microfacet;
            break;
        default:
            assert(false);
    }
    return *this;
}
