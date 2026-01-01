//
// Created by steinraf on 26.02.25.
//

#include "../geometry/intersection.cuh"
#include "../rng/sampler.cuh"
#include "bsdf.cuh"

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

CPU_GPU Color BSDF::eval(const BSDFQueryRecord &query) const noexcept {
    return material.eval(texture, query);
}
CPU_GPU float BSDF::pdf(const BSDFQueryRecord &query) const noexcept {
    return material.pdf(query);
}

BSDF::BSDF(Material material, Texture texture) : material(material), texture(texture) {
}
__device__ Color BSDF::sample(BSDFQueryRecord &bsdfQueryRecord, const Vec2f &randomSample) const noexcept {
    switch(material.type) {
        case MaterialType::DIFFUSE:
            if(Frame::cosTheta(bsdfQueryRecord.wIn) <= 0) {
                bsdfQueryRecord.wOut = Vec3f{0.0f, 0.0f, 1.0f};
                return Color::Zero();
            }

            bsdfQueryRecord.measure = EMeasure::ESolidAngle;

            bsdfQueryRecord.wOut = sample::squareToCosineHemisphere(randomSample);

            bsdfQueryRecord.eta = 1.0f;

            return texture.eval(bsdfQueryRecord.uv);
        case MaterialType::SPECULAR:
            if(Frame::cosTheta(bsdfQueryRecord.wIn) <= 0)
                return Color::Zero();

            assert(isfinite(bsdfQueryRecord.wIn[0]));
            assert(isfinite(bsdfQueryRecord.wIn[1]));
            assert(isfinite(bsdfQueryRecord.wIn[2]));

            bsdfQueryRecord.wOut = Vec3f{
                    -bsdfQueryRecord.wIn[0],
                    -bsdfQueryRecord.wIn[1],
                    bsdfQueryRecord.wIn[2]};
            bsdfQueryRecord.measure = EMeasure::EDiscrete;

            bsdfQueryRecord.eta = 1.0f;

            return Color::Ones();
        case MaterialType::DIELECTRIC:
            return [&]() -> Color {
                float extIOR = material.iorExterior(), intIOR = material.iorInterior(), cosThetaI = Frame::cosTheta(bsdfQueryRecord.wIn);
                Vec3f normal = Vec3f::UnitZ();
                if(cosThetaI < 0) {
                    std::swap(extIOR, intIOR);
                    cosThetaI *= -1;
                    normal *= -1;
                }


                const float fresnelCoeff = fresnel(cosThetaI, extIOR, intIOR);

                bsdfQueryRecord.measure = EMeasure::EDiscrete;

                if(randomSample[0] < fresnelCoeff) {
                    bsdfQueryRecord.eta = 1.f;

                    assert(isfinite(bsdfQueryRecord.wIn[0]));
                    assert(isfinite(bsdfQueryRecord.wIn[1]));
                    assert(isfinite(bsdfQueryRecord.wIn[2]));

                    bsdfQueryRecord.wOut = Vec3f(
                            -bsdfQueryRecord.wIn[0],
                            -bsdfQueryRecord.wIn[1],
                            bsdfQueryRecord.wIn[2]);


                    return Color::Ones();

                } else {
                    bsdfQueryRecord.eta = extIOR / intIOR;

                    bsdfQueryRecord.wOut =
                            -bsdfQueryRecord.eta * (bsdfQueryRecord.wIn - (bsdfQueryRecord.wIn.dot(normal) * normal)) - normal * sqrt(1 - bsdfQueryRecord.eta * bsdfQueryRecord.eta * (1 - bsdfQueryRecord.wIn[2] * bsdfQueryRecord.wIn[2]));

                    assert(isfinite(bsdfQueryRecord.wOut[0]));
                    assert(isfinite(bsdfQueryRecord.wOut[1]));
                    assert(isfinite(bsdfQueryRecord.wOut[2]));

                    return bsdfQueryRecord.eta * bsdfQueryRecord.eta * Color::Ones();
                }
            }();

        case MaterialType::MICROFACET:
            assert(false);
            return Color::Zero();
    }
}
CPU_GPU Color BSDF::evalTexture(const Vec2f &uv) const noexcept {
    return texture.eval(uv);
}
CPU_GPU bool BSDF::hasZeroTexture() const noexcept {
    return texture.type == TextureType::CONSTANT && texture.getConstant() == Color::Zero();
}
CPU_GPU void BSDF::setUnitTexture() noexcept {
    texture = Texture::ONES();
}
