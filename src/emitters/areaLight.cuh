//
// Created by steinraf on 28/11/22.
//

#pragma once

#include "../common.h"


#include "../../src_old/utility/ray.h"
#include "../../src_old/shapes/triangle.h"





class AreaLight {
public:
    CPU_GPU_CONSTEXPR explicit AreaLight(const Color3f &radiance) noexcept
        : radiance(radiance), blas(nullptr) {
    }

    AreaLight() = default;

    CPU_GPU_CONSTEXPR void setBlas(const class BLAS *newBlas) {
        blas = newBlas;
    }


    [[nodiscard]] CPU_GPU Color3f eval(const EmitterQueryRecord &emitterQueryRecord) const noexcept;

    [[nodiscard]] CPU_GPU float pdf(const EmitterQueryRecord &emitterQueryRecord) const noexcept;


    [[nodiscard]] CPU_GPU Color3f
    sample(EmitterQueryRecord &emitterQueryRecord, const Vector3f &sample) const noexcept;


    [[nodiscard]] CPU_GPU_CONSTEXPR bool isEmitter() const noexcept {
        return !radiance.isZero();
    }

    //TODO add texture as radiance option
    Color3f radiance;
    const BLAS *blas = nullptr;
};

