//
// Created by steinraf on 20/12/22. Ported from code written by joluther.
//

#pragma once

#include "../bsdf.h"
#include "../utility/vector.cuh"

struct PhaseFunctionQueryRecord {
    Vector3f wi;
    Vector3f wo;
    float eta;

    EMeasure measure;

    Vector3f p;


    __device__ PhaseFunctionQueryRecord(const Vector3f &wi)
        : wi(wi), measure(EUnknownMeasure) {

    }

    __device__ PhaseFunctionQueryRecord(const Vector3f &wi,
                             const Vector3f &wo, EMeasure measure)
        : wi(wi), wo(wo), measure(measure) {

    }

};


class IsotropicPhaseFunction {

public:
    [[nodiscard]] static __host__ __device__ Color3f sample(PhaseFunctionQueryRecord &bRec, const Vector2f &sample) {
        bRec.wo = Warp::squareToUniformSphere(sample);
        bRec.measure = ESolidAngle;
        bRec.eta = 1.0f;
        return Color3f{1.0};
    };

    [[nodiscard]] static __host__ __device__ Color3f eval(const PhaseFunctionQueryRecord &bRec) {
        return 0.25 * Color3f{1.00} * M_1_PI;
    };

    [[nodiscard]] __host__ __device__ float pdf(const PhaseFunctionQueryRecord &bRec) const {
        return 0.25f * M_1_PIf;
    };

    //Added for interchangeable nori syntax
    __device__ IsotropicPhaseFunction* operator->(){
        return this;
    }

    __device__ const IsotropicPhaseFunction* operator->() const{
        return this;
    }
};
