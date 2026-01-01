//
// Created by steinraf on 05.02.25.
//

#include "../acceleration/multibvh.cuh"
#include "../camera/camera.cuh"
#include "../emitter/emitter.cuh"
#include "../geometry/intersection.cuh"
#include "../gui/viewport.cuh"
#include "../material/bsdf.cuh"
#include "../rng/sampler.cuh"
#include "integrators.cuh"

#define mis render_kern

__global__ void mas(TLAS *tlas, Texture envmap, FeatureBuffer *buffer,
                    Camera camera, curandState *rngStates,
                    unsigned int width, unsigned int height, int spp) {


    constexpr int maxBounces = 16;


    for(size_t pixelIndex = blockIdx.x * blockDim.x + threadIdx.x;
        pixelIndex < width * height; pixelIndex += blockDim.x * gridDim.x) {
        size_t x = pixelIndex % width, y = pixelIndex / width;

        const float u = float(x) / float(width);
        const float v = float(y) / float(height);

        Sampler sampler{&rngStates[pixelIndex]};


        auto screenPos = Vec3f{u, v, 0.0f};

        const Vec3f backgroundColor =
                0.0f * Vec3f{1.0, 1.0, 1.0};

        Vec3f totalColor{0.0, 0.0, 0.0};

        for(int sample = 0; sample < spp; ++sample) {


            Intersection intersection;
            Ray currentRay = camera.getRay(u, v, sampler);
            auto color = Vec3f{0.0, 0.0, 0.0};
            auto t = Vec3f{1.0, 1.0, 1.0};
            float etaScale = 1.0;// TODO Changes to Russian Roulette due to
                                 // indices of refraction

            int numBounces = 0;

            while(true) {
                if(!tlas->intersect(currentRay, intersection, false)) {

                    color.array() += t.array() * envmap.eval(currentRay).array();
                    if(numBounces == 0) {
                        buffer->normal[pixelIndex].addElement(Vec3f{0.0, 0.0, 0.0});
                        buffer->position[pixelIndex].addElement(Vec3f{0.0, 0.0, 0.0});
                        buffer->albedo[pixelIndex].addElement(Vec3f{0.0, 0.0, 0.0});
                        buffer->uv[pixelIndex].addElement(Vec3f{0.0, 0.0, 0.0});
                    }
                    break;
                } else if(numBounces == 0) {

                    BSDFQueryRecord bsdfQueryRecord{intersection.shFrame.toLocal(-currentRay.dir)};
                    bsdfQueryRecord.measure = EMeasure::ESolidAngle;
                    bsdfQueryRecord.uv = intersection.uv;
                    auto bsdfSample = intersection.meshf->bsdf.sample(bsdfQueryRecord, sampler.getSample2D());

                    buffer->normal[pixelIndex].addElement(intersection.shFrame.n);
                    buffer->position[pixelIndex].addElement(intersection.point);
                    buffer->albedo[pixelIndex].addElement(bsdfSample);
                    buffer->uv[pixelIndex].addElement(Vec3f{intersection.uv[0], intersection.uv[1], 0.0});
                }

                constexpr int maxEnvSamples = 0;
                for(int envSamples = 0; envSamples < maxEnvSamples; ++envSamples) {
                    EmitterQueryRecord envMapEQR{intersection.point};

                    const Color envMapEMSSample = envmap.sample(envMapEQR, sampler.getSample3D());

                    if(!tlas->intersect(envMapEQR.shadowRay)) {

                        BSDFQueryRecord bsdfQueryRecord{
                                intersection.shFrame.toLocal(-currentRay.dir),
                                intersection.shFrame.toLocal(envMapEQR.wIn),
                                EMeasure::ESolidAngle};
                        bsdfQueryRecord.measure = EMeasure::ESolidAngle;
                        bsdfQueryRecord.uv = intersection.uv;

                        const float tempPDF = envmap.pdf(envMapEQR);

                        color.array() += envMapEMSSample.array() * t.array() * intersection.meshf->bsdf.eval(bsdfQueryRecord).array() * Frame::cosTheta(intersection.shFrame.toLocal(envMapEQR.wIn)) * tempPDF / (intersection.meshf->bsdf.pdf(bsdfQueryRecord) + tempPDF) / maxEnvSamples;
                    }
                }

                if(intersection.isEmitter()) {
                    if(intersection.shFrame.n.dot(currentRay.dir) < 0) {
                        color.array() += t.array() * intersection.emitter->radiance.array();
                    }
                }


                // roulette
                float successProbability = min(t.maxCoeff() * etaScale, 0.99f);
                if(sampler.getSample1D() > successProbability ||
                   numBounces > maxBounces) {
                    break;
                }

                t.array() /= successProbability;

                BSDFQueryRecord bsdfQueryRecord{intersection.shFrame.toLocal(-currentRay.dir)};
                bsdfQueryRecord.measure = EMeasure::ESolidAngle;
                bsdfQueryRecord.uv = intersection.uv;

                auto bsdfSample = intersection.meshf->bsdf.sample(bsdfQueryRecord, sampler.getSample2D());


                t.array() *= bsdfSample.array();// BSDF

                currentRay = Ray{
                        intersection.point,
                        intersection.shFrame.toWorld(bsdfQueryRecord.wOut)};

                ++numBounces;
            }

            totalColor += color;
        }


        totalColor /= float(spp);


        buffer->color[pixelIndex].addElement(totalColor);
    }
}

__global__ void bvhDebugger(TLAS *tlas, Texture envmap, FeatureBuffer *buffer,
                            Camera camera, curandState *rngStates,
                            unsigned int width, unsigned int height, int spp) {

    for(size_t pixelIndex = blockIdx.x * blockDim.x + threadIdx.x;
        pixelIndex < width * height; pixelIndex += blockDim.x * gridDim.x) {
        size_t x = pixelIndex % width, y = pixelIndex / width;

        const float u = float(x) / float(width);
        const float v = float(y) / float(height);

        Sampler sampler{&rngStates[pixelIndex]};

        EmitterQueryRecord eqr{camera.getRay(u, v, sampler).origin};

        auto screenPos = Vec3f{u, v, 0.0f};

        Vec3f totalColor{0.0, 0.0, 0.0};

        for(int sample = 0; sample < spp; ++sample) {


            Ray currentRay = camera.getRay(u, v, sampler);
            auto color = Vec3f{0.0, 0.0, 0.0};

            Intersection its;

            if(!tlas->intersect(currentRay, its)) {
                color.array() = envmap.eval(currentRay).array();
            } else {
                if(its.triangle) {
                    color.array() = Vec3f{its.uv[0], its.uv[1], 0.5};
                } else {
                    color.array() = Color::Ones();
                }
            }

            totalColor += color;
        }


        totalColor /= float(spp);


        buffer->color[pixelIndex].addElement(totalColor);
    }
}


__global__ void mis(TLAS *tlas, Texture envmap, FeatureBuffer *buffer,
                    Camera camera, curandState *rngStates,
                    unsigned int width, unsigned int height, int spp) {


    constexpr int maxBounces = 4;


    for(size_t pixelIndex = blockIdx.x * blockDim.x + threadIdx.x;
        pixelIndex < width * height; pixelIndex += blockDim.x * gridDim.x) {
        size_t x = pixelIndex % width, y = pixelIndex / width;

        const float u = float(x) / float(width);
        const float v = float(y) / float(height);

        Sampler sampler{&rngStates[pixelIndex]};

        EmitterQueryRecord eqr{camera.getRay(u, v, sampler).origin};
        //
        //        for(int i = 0; i < width * height; ++i) {
        //            if(i == pixelIndex and i % 100 == 0) {
        //                Vec3f sample = envmap.sample(eqr, sampler.getSample3D());
        //                auto mapSample = [](float x) {
        //                    return std::clamp(x, 0.f, 1.f);
        //                };
        //                for(int i = 0; i < 3; ++i) sample[i] = mapSample(sample[i] * eqr.pdf);
        //                auto idx = eqr.idx;
        //                buffer->color[idx].addElement(sample);
        //            }
        //            __syncthreads();
        //        }

        //        Vec3f sample = envmap.sample(eqr, sampler.getSample3D());
        //        auto mapSample = [](float x) {
        //            return std::clamp(x, 0.f, 1.f);
        //        };
        //        for(int i = 0; i < 3; ++i) sample[i] = mapSample(1000 * eqr.pdf);
        //        auto idx = eqr.idx;
        //        buffer->color[idx].addElement(sample);// race condition here
        //
        //        continue;

        auto screenPos = Vec3f{u, v, 0.0f};

        const Vec3f backgroundColor =
                0.0f * Vec3f{1.0, 1.0, 1.0};

        Vec3f totalColor{0.0, 0.0, 0.0};

        float wMat = 1.0f;

        for(int sample = 0; sample < spp; ++sample) {


            Intersection intersection;
            Ray currentRay = camera.getRay(u, v, sampler);
            auto color = Vec3f{0.0, 0.0, 0.0};
            auto t = Vec3f{1.0, 1.0, 1.0};
            float etaScale = 1.0;// TODO Changes to Russian Roulette due to
                                 // indices of refraction

            int numBounces = 0;

            while(true) {
                if(!tlas->intersect(currentRay, intersection, false)) {

                    color.array() += t.array() * envmap.eval(currentRay).array();
                    if(numBounces == 0) {
                        buffer->normal[pixelIndex].addElement(Vec3f{0.0, 0.0, 0.0});
                        buffer->position[pixelIndex].addElement(Vec3f{0.0, 0.0, 0.0});
                        buffer->albedo[pixelIndex].addElement(Vec3f{0.0, 0.0, 0.0});
                        buffer->uv[pixelIndex].addElement(Vec3f{0.0, 0.0, 0.0});
                    }
                    break;
                } else if(numBounces == 0) {

                    BSDFQueryRecord bsdfQueryRecord{intersection.shFrame.toLocal(-currentRay.dir)};
                    bsdfQueryRecord.measure = EMeasure::ESolidAngle;
                    bsdfQueryRecord.uv = intersection.uv;
                    auto bsdfSample = intersection.meshf->bsdf.sample(bsdfQueryRecord, sampler.getSample2D());

                    buffer->normal[pixelIndex].addElement(intersection.shFrame.n);
                    buffer->position[pixelIndex].addElement(intersection.point);
                    buffer->albedo[pixelIndex].addElement(bsdfSample);
                    buffer->uv[pixelIndex].addElement(Vec3f{intersection.uv[0], intersection.uv[1], 0.0});
                }

                if(tlas->containsEmitters()) {
                    const auto *light = tlas->getRandomEmitter(sampler.getSample1D());

                    if(light != intersection.emitter) {
                        assert(light);
                        EmitterQueryRecord emitterQueryRecord{intersection.point};


                        const Color lightSample = light->sample(emitterQueryRecord, sampler.getSample3D()) * tlas->getEmitterCount();

                        if(!tlas->intersect(emitterQueryRecord.shadowRay)) {

                            BSDFQueryRecord bsdfQueryRecord{
                                    intersection.shFrame.toLocal(-currentRay.dir),
                                    intersection.shFrame.toLocal(emitterQueryRecord.wIn),
                                    EMeasure::ESolidAngle};
                            bsdfQueryRecord.measure = EMeasure::ESolidAngle;
                            bsdfQueryRecord.uv = intersection.uv;

                            const float tempPDF = light->pdf(emitterQueryRecord);

                            color.array() += t.array() *
                                             lightSample.array() *
                                             intersection.meshf->bsdf.eval(bsdfQueryRecord).array() *
                                             Frame::cosTheta(intersection.shFrame.toLocal(emitterQueryRecord.wIn)) *
                                             tempPDF /
                                             (intersection.meshf->bsdf.pdf(bsdfQueryRecord) + tempPDF);
                        }
                    }
                }


                constexpr int maxEnvSamples = 0;
                for(int envSamples = 0; envSamples < maxEnvSamples; ++envSamples) {
                    EmitterQueryRecord envMapEQR{intersection.point};

                    const Color envMapEMSSample = envmap.sample(envMapEQR, sampler.getSample3D());


                    if(!tlas->intersect(envMapEQR.shadowRay)) {

                        BSDFQueryRecord bsdfQueryRecord{
                                intersection.shFrame.toLocal(-currentRay.dir),
                                intersection.shFrame.toLocal(envMapEQR.wIn),
                                EMeasure::ESolidAngle};
                        bsdfQueryRecord.measure = EMeasure::ESolidAngle;
                        bsdfQueryRecord.uv = intersection.uv;

                        const float tempPDF = envmap.pdf(envMapEQR);

                        color.array() += t.array() *
                                         envMapEMSSample.array() *
                                         intersection.meshf->bsdf.eval(bsdfQueryRecord).array() *
                                         Frame::cosTheta(intersection.shFrame.toLocal(envMapEQR.wIn)) *
                                         tempPDF /
                                         (intersection.meshf->bsdf.pdf(bsdfQueryRecord) + tempPDF) /
                                         maxEnvSamples;
                    }
                }

                if(intersection.isEmitter()) {
                    color.array() += t.array() * wMat * intersection.emitter->eval({currentRay.origin, intersection.point, intersection.shFrame.n, intersection.uv}).array();
                    //                    break;// assumes all emitters are perfect black bodies
                }


                // roulette
                float successProbability = min(t.maxCoeff() * etaScale, 0.99f);
                if(sampler.getSample1D() > successProbability ||
                   numBounces > maxBounces) {
                    break;
                }

                t.array() /= successProbability;

                BSDFQueryRecord bsdfQueryRecord{intersection.shFrame.toLocal(-currentRay.dir)};
                bsdfQueryRecord.measure = EMeasure::ESolidAngle;
                bsdfQueryRecord.uv = intersection.uv;

                t.array() *= intersection.meshf->bsdf.sample(bsdfQueryRecord, sampler.getSample2D()).array();// BSDF

                currentRay = Ray{
                        intersection.point,
                        intersection.shFrame.toWorld(bsdfQueryRecord.wOut)};

                const float masPDF = intersection.meshf->bsdf.pdf(bsdfQueryRecord);

                Intersection masEmitterIntersection;
                if(!tlas->intersect(currentRay, masEmitterIntersection, true)) {
                    color.array() += t.array() * envmap.eval(currentRay).array();
                    break;
                }

                if(masEmitterIntersection.isEmitter()) {
                    const float emsPDF = masEmitterIntersection.emitter->pdf({currentRay.origin,
                                                                              masEmitterIntersection.point,
                                                                              masEmitterIntersection.shFrame.n,
                                                                              masEmitterIntersection.uv});
                    wMat = (masPDF + emsPDF > 0) ? masPDF / (masPDF + emsPDF) : masPDF;
                }

                if(bsdfQueryRecord.measure == EMeasure::EDiscrete) {
                    wMat = 1.0f;
                }

                ++numBounces;
            }

            totalColor += color;
        }


        totalColor /= float(spp);


        buffer->color[pixelIndex].addElement(totalColor);
    }
}


__global__ void bufferToSurface(cudaSurfaceObject_t surface, FeatureBuffer *buffer, unsigned int width, unsigned int height) {
    for(size_t pixelIndex = blockIdx.x * blockDim.x + threadIdx.x;
        pixelIndex < width * height; pixelIndex += blockDim.x * gridDim.x) {
        size_t x = width - 1 - pixelIndex % width, y = pixelIndex / width;

        auto color = tonemap(buffer->color[pixelIndex].getMean());

        auto toChar = [](float x) {
            return static_cast<unsigned char>(std::clamp(x * 255.f, 0.f, 255.f));
        };

        uchar4 color4 = make_uchar4(toChar(color[0]),
                                    toChar(color[1]),
                                    toChar(color[2]),
                                    255);


        surf2Dwrite(color4, surface, x * sizeof(uchar4), y);
    }
}


__device__ Vec3f tonemap(Vec3f color) {

    auto gammaCorrect = [] __device__(float x) {
        auto clamp = [] __device__(float x, float min, float max) {
            if(std::isinf(x) || std::isnan(x)) return min;
            return x < min ? min : (x > max ? max : x);
        };

        if(x <= 0.0031308f) return clamp(12.92f * x, 0.f, 1.f);
        return clamp(1.055f * std::pow(x, 1.f / 2.4f) - 0.055f, 0.f,
                     1.f);
    };

    for(int c = 0; c < 3; ++c) {
        color[c] = gammaCorrect(color[c]);
        assert(color[c] <= 1.0);
    }
    return color;
}