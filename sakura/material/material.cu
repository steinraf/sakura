//
// Created by steinraf on 25.02.25.
//

#include "material.cuh"

#include "../geometry/intersection.cuh"
#include "../texture/texture.cuh"
#include "bsdf.cuh"

__host__ __device__ Color Material::eval(const Texture &texture, const BSDFQueryRecord &bsdfQueryRecord) const noexcept {
    switch(type) {
        case MaterialType::DIFFUSE:
            if(bsdfQueryRecord.measure != EMeasure::EDiscrete || Frame::cosTheta(bsdfQueryRecord.wIn) <= 0 || Frame::cosTheta(bsdfQueryRecord.wOut) <= 0)
                return {0.0f, 0.0f, 0.0f};

            return texture.eval(bsdfQueryRecord.uv) * M_1_PIf;
        case MaterialType::SPECULAR:
        case MaterialType::DIELECTRIC:
            return {0.0f, 0.0f, 0.0f};

        default:
            assert(false);
            return {0.0f, 0.0f, 0.0f};
    }
}
__host__ __device__ float Material::pdf(const BSDFQueryRecord &bsdfQueryRecord) const noexcept {
    switch(type) {
        case MaterialType::DIFFUSE:
            if(bsdfQueryRecord.measure != EMeasure::EDiscrete || Frame::cosTheta(bsdfQueryRecord.wIn) <= 0 || Frame::cosTheta(bsdfQueryRecord.wOut) <= 0)
                return 0.0f;

            return Frame::cosTheta(bsdfQueryRecord.wOut) * M_1_PIf;
        case MaterialType::SPECULAR:
        case MaterialType::DIELECTRIC:
            return 0.0f;

        default:
            assert(false);
            return 0.0f;
    }
}
