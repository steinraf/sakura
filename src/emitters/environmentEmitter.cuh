//
// Created by steinraf on 07/12/22.
//
#pragma once


#include "../common.h"

#include "../../src_old/textures/imageTexture.h"

#include "../../src_old/utility/warp.h"

#include "../../src/emitters/queryRecord.cuh"


#define ENVIRONMENT_EPSILON 1e-6f

class EnvironmentEmitter {
public:
    Texture texture;

    explicit CPU_GPU constexpr EnvironmentEmitter(Texture texture) noexcept
        : texture(texture) {

    }
    
    [[nodiscard]] CPU_GPU constexpr Color3f eval(const Ray3f &ray) const noexcept{

        assert(ray.getDirection().norm() != 0.f);
        const Vector3f dir = ray.getDirection().normalized();

        const float u = atan2(dir[0], -dir[2]) * 0.5f * M_1_PIf;
        const float v = std::clamp(acos(-dir[1]) * M_1_PIf, -1.f, 1.f);

        if(!::isfinite(u) || !::isfinite(v)){
#ifndef NDEBUG
            printf("UV infinite for dir (%f, %f, %f)\n", ray.getDirection()[0], ray.getDirection()[1], ray.getDirection()[2]);
#endif
            return Vector3f{0.f};
        }
        assert(isfinite(u) && ::isfinite(v));
        return texture.eval(Vector2f{(u < 0) ? (u + 1) : u, v});
    }

    [[nodiscard]] CPU_GPU constexpr float pdf(const EmitterQueryRecord &emitterQueryRecord) const noexcept{ //TODO find where sin factor needs to be
        const float pdf = texture.pdf(emitterQueryRecord.idx);
        if(pdf == 0 || sin(M_PIf*emitterQueryRecord.uv[1]) == 0) return ENVIRONMENT_EPSILON ;
        return texture.pdf(emitterQueryRecord.idx)  *M_1_PIf*M_1_PIf/(2*sin(M_PIf*emitterQueryRecord.uv[1]));
    }

    [[nodiscard]] CPU_GPU constexpr Color3f sample(EmitterQueryRecord &emitterQueryRecord, const Vector3f &sample) const noexcept{
        //TODO check if this actually needs 3d sample input or if 1d is sufficient
        if(!texture.deviceCDF)
            return texture.eval(Vector2f{});


        const size_t idx = Warp::sampleCDF(sample[2], texture.deviceCDF, texture.deviceCDF + (texture.width * texture.height - 1));

        emitterQueryRecord.idx = idx;


        const float u = (idx % texture.width)*1.f/texture.width;
        const float v = (idx / texture.width)*1.f/texture.height;

        assert(u >= 0 && u <= 1 && v >= 0 && v <= 1);

        const Vector3f warpSample{Warp::squareToUniformSphere(Vector2f{u, v})};

        const Vector3f dirSample = warpSample;

        emitterQueryRecord.shadowRay = Ray3f{
                emitterQueryRecord.p,
                dirSample,
        };

        emitterQueryRecord.p = emitterQueryRecord.ref + 100000*dirSample; //TODO change to infinity maybe
        emitterQueryRecord.wi = -dirSample;


        emitterQueryRecord.uv = Vector2f{u, v};

        //https://cs184.eecs.berkeley.edu/sp18/article/25
        const float pdf = this->pdf(emitterQueryRecord);
        emitterQueryRecord.pdf = pdf;
        assert(pdf > 0);
        return texture.eval(emitterQueryRecord.uv)/pdf;
    }

};

