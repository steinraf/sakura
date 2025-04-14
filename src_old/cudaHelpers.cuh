//
// Created by steinraf on 19/08/22.
//

#pragma once

#include <curand_kernel.h>

#include "../src/camera/camera.cuh"
#include "acceleration/bvh.h"
#include "utility/ray.h"
#include "utility/warp.h"
#include <cuda/std/limits>


#include "../../src/denoise/featureBuffer.cuh"




namespace cudaHelpers {


    __device__ bool initIndices(int &i, int &j, int &pixelIndex, const int width, const int height) noexcept;


    __global__ void initRng(int width, int height, curandState *randState);


    // The findSplit, delta and determineRange are taken from here
    // https://developer.nvidia.com/blog/thinking-parallel-part-iii-tree-construction-gpu/
    // https://github.com/nolmoonen/cuda-lbvh/blob/main/src/build.cu

    __device__ int findSplit(const uint32_t *mortonCodes, int first, int last, int numPrimitives);

    __device__ int delta(int a, int b, unsigned int n, const unsigned int *c, unsigned int ka);

    __device__ std::pair<int, int> determineRange(const uint32_t *mortonCodes, int numPrimitives, int i);


    __global__ void constructBVH(AccelerationNode *bvhNodes, Triangle *primitives, const uint32_t *mortonCodes, int numPrimitives);

    __device__ AABB getBoundingBox(AccelerationNode *root) noexcept;

    __global__ void computeBVHBoundingBoxes(AccelerationNode *bvhNodes) ;

    __global__ void initBVH(BLAS *bvh, AccelerationNode *bvhTotalNodes, float totalArea, const float *cdf,
            size_t numPrimitives, AreaLight *emitter, BSDF bsdf, Texture normalMap);

    __global__ void freeVariables();

//     CPU_GPU_INLINE Color3f DirectMAS(const Ray3f &ray, TLAS *scene, Sampler &sampler) noexcept {
//        Intersection its;
//        if(!scene->rayIntersect(ray, its))
//            return Color3f{0.f};
//
//
//        Color3f sample{0.f};
//
//        if(its.mesh->isEmitter()) {
//            sample = its.mesh->getEmitter()->eval({ray.o, its.p, its.shFrame.n, its.uv});
//        }
//
//
//        BSDFQueryRecord bsdfQueryRecord{
//                its.shFrame.toLocal(-ray.d)};
//        bsdfQueryRecord.measure = ESolidAngle;
//        bsdfQueryRecord.uv = its.uv;
//
//        auto bsdfSample = its.mesh->getBSDF()->sample(bsdfQueryRecord, sampler.getSample2D());
//
//        Ray3f newRay = {
//                its.p,
//                its.shFrame.toWorld(bsdfQueryRecord.wo)};
//
//        Intersection emitterIntersect;
//
//        if(scene->rayIntersect(newRay, emitterIntersect) && emitterIntersect.mesh->isEmitter()) {
//
//            const auto &emitter = emitterIntersect.mesh->getEmitter();
//
//            EmitterQueryRecord emitterQueryRecord{
//                    ray.o,
//                    its.p,
//                    its.shFrame.n,
//                    its.uv
//            };
//
//            sample += emitter->eval(emitterQueryRecord) * bsdfSample;
//        }
//
//        return sample;
//    }
//
//    CPU_GPU_CONSTEXPR Color3f DirectMIS(const Ray3f &ray, TLAS *scene, Sampler &sampler) {
//        Intersection its;
//        if(!scene->rayIntersect(ray, its))
//            return Color3f{0.f};
//
//        Color3f sample{0.f};
//
//        if(its.mesh->isEmitter())
//            sample = its.mesh->getEmitter()->eval({ray.o, its.p, its.shFrame.n, its.uv});
//
//
//        const auto &emsLight = scene->getRandomEmitter(sampler.getSample1D());
//
//
//        EmitterQueryRecord emsEmitterQueryRecord{its.p};
//
//
//        const auto &currentLight = emsLight->sample(emsEmitterQueryRecord, sampler.getSample3D());
//
//
//        BSDFQueryRecord emsBSDFRec{
//                its.shFrame.toLocal(-ray.d),
//                its.shFrame.toLocal(emsEmitterQueryRecord.wi),
//                ESolidAngle};
//
//        emsBSDFRec.uv = its.uv;
//
//
//        Color3f emsSample{0.f};
//        float emsWeight = 1;
//
//
//        if(!scene->rayIntersect(emsEmitterQueryRecord.shadowRay)) {
//            emsWeight = emsLight->pdf(emsEmitterQueryRecord) /
//                        (emsLight->pdf(emsEmitterQueryRecord) + its.mesh->getBSDF()->pdf(emsBSDFRec));
//
//            emsSample = its.mesh->getBSDF()->eval(emsBSDFRec) * currentLight * std::abs(its.shFrame.n.dot(emsEmitterQueryRecord.shadowRay.d)) * scene->numEmitters;
//        }
//
//        BSDFQueryRecord masBSDFQueryRecord{
//                its.shFrame.toLocal(-ray.d)};
//
//        masBSDFQueryRecord.uv = its.uv;
//
//        auto masBSDFSample = its.mesh->getBSDF()->sample(masBSDFQueryRecord, sampler.getSample2D());
//
//        Ray3f newMASRay{
//                its.p,
//                its.shFrame.toWorld(masBSDFQueryRecord.wo)};
//
//        Intersection masEmitterIntersect;
//        Color3f masSample{0.f};
//        float masWeight = 1;
//
//
//        if(scene->rayIntersect(newMASRay, masEmitterIntersect) && masEmitterIntersect.mesh->isEmitter()) {
//
//            EmitterQueryRecord masEmitterQueryRecord{
//                    its.p,
//                    masEmitterIntersect.p,
//                    masEmitterIntersect.shFrame.n,
//                    masEmitterIntersect.uv
//            };
//
//            const auto &masEmitter = masEmitterIntersect.mesh->getEmitter();
//
//            masWeight = its.mesh->getBSDF()->pdf(masBSDFQueryRecord) /
//                        (masEmitter->pdf(masEmitterQueryRecord) + its.mesh->getBSDF()->pdf(masBSDFQueryRecord));
//
//            masSample = masEmitter->eval(masEmitterQueryRecord) * masBSDFSample;
//        }
//
//        return sample + emsWeight * emsSample + masWeight * masSample;
//    }
//
//    CPU_GPU_CONSTEXPR Color3f PathMAS(const Ray3f &ray, TLAS *scene, int maxRayDepth, Sampler &sampler, FeatureBuffer &featureBuffer) noexcept {
//        Intersection its;
//
//
//        Color3f Li{0.f}, t{1.f};
//
//        Ray3f currentRay = ray;
//
//        int numBounces = 0;
//
//        while(true) {
//
//            if(!scene->rayIntersect(currentRay, its))
//                return Li;
//
//
//            if(its.mesh->isEmitter())
//                Li += t * its.mesh->getEmitter()->eval({currentRay.o, its.p, its.shFrame.n, its.uv});
//
//            float successProbability = fmin(t.maxCoeff(), 0.99f);
//            //                if((++numBounces > 3) && sampler->next1D() > successProbability)
//            if(sampler.getSample1D() >= successProbability || ++numBounces > maxRayDepth)
//                return Li;
//
//            t /= successProbability;
//
//            BSDFQueryRecord bsdfQueryRecord{
//                    its.shFrame.toLocal(-currentRay.d)};
//            bsdfQueryRecord.measure = ESolidAngle;
//            bsdfQueryRecord.uv = its.uv;
//
//            const auto bsdfSample = its.mesh->getBSDF()->sample(bsdfQueryRecord, sampler.getSample2D());
//
//            t *= bsdfSample;
//
//            currentRay = {
//                    its.p,
//                    its.shFrame.toWorld(bsdfQueryRecord.wo)};
//        }
//    }
//
//    CPU_GPU_CONSTEXPR Color3f PathMIS(const Ray3f &ray, TLAS *scene, int maxRayDepth, Sampler &sampler,
//                                         FeatureBuffer *featureBuffer, size_t fbIndex) noexcept {
//        Intersection its;
//
//
//        Color3f Li{0.f}, t{1.f};
//
//        Ray3f currentRay = ray;
//
//        int numBounces = 0;
//
//        float wMat = 1.0f;
//
//
//        while(true) {
//            assert(currentRay.getDirection().norm() != 0.f);
//            assert(currentRay.getDirection().isValid());
//
//            if(!scene->rayIntersect(currentRay, its)) {
//                if(t.norm() > EPSILON)
//                    return Li + t * scene->environmentEmitter.eval(currentRay);
//                else
//                    return Li;
//            }
//
//            if(numBounces == 0) {
//                featureBuffer->position[fbIndex].addElement(its.p);
//                featureBuffer->normal[fbIndex].addElement(its.mesh->getBSDF().material == Material::DIELECTRIC ? Vector3f{1.f} : its.shFrame.n);
//                featureBuffer->albedo[fbIndex].addElement(its.mesh->getBSDF().material == Material::DIELECTRIC ? Vector3f{1.f} : its.mesh->getBSDF()->getAlbedo(its.uv));
//            }
//
//
//            const auto *light = scene->getRandomEmitter(sampler.getSample1D());
//
//            EmitterQueryRecord emitterQueryRecord{
//                    its.p};
//
//            const Color3f emsSample = light->sample(emitterQueryRecord, sampler.getSample3D()) * scene->numEmitters;
//
//            if(!scene->rayIntersect(emitterQueryRecord.shadowRay)) {
//
//                BSDFQueryRecord bsdfQueryRecord{
//                        its.shFrame.toLocal(-currentRay.d),
//                        its.shFrame.toLocal(emitterQueryRecord.wi),
//                        ESolidAngle};
//                bsdfQueryRecord.measure = ESolidAngle;
//                bsdfQueryRecord.uv = its.uv;
//
//                Li += emsSample * its.mesh->getBSDF()->eval(bsdfQueryRecord) * Frame::cosTheta(its.shFrame.toLocal(emitterQueryRecord.wi)) * light->pdf(emitterQueryRecord) /
//                      (its.mesh->getBSDF()->pdf(bsdfQueryRecord) + light->pdf(emitterQueryRecord)) * t;
//            }
//
////            EmitterQueryRecord envEQR{its.p};
////            const Color3f envSample = scene->environmentEmitter.sample(envEQR, sampler.getSample3D());
////            if(!scene->rayIntersect(envEQR.shadowRay)){
////                Li += t * wMat * envSample;
////            }
//
//            if(its.mesh->isEmitter())
//                Li += t * wMat * its.mesh->getEmitter()->eval({currentRay.o, its.p, its.shFrame.n, its.uv});
//
//            float successProbability = fmin(t.maxCoeff(), 0.99f);
//            //                if((++numBounces > 3) && sampler->next1D() > successProbability)
//            if(sampler.getSample1D() >= successProbability || numBounces > maxRayDepth){
////                if(t.norm() > EPSILON)
////                    return Li + t * scene->environmentEmitter.eval(currentRay);
////                else
//                    return Li;
//            }
//
//            t /= successProbability;
//
//            BSDFQueryRecord bsdfQueryRecord{
//                    its.shFrame.toLocal(-currentRay.d)};
//            bsdfQueryRecord.measure = ESolidAngle;
//            bsdfQueryRecord.uv = its.uv;
//
//            t *= its.mesh->getBSDF()->sample(bsdfQueryRecord, sampler.getSample2D());
//
//            currentRay = {
//                    its.p,
//                    its.shFrame.toWorld(bsdfQueryRecord.wo)
//            };
//
//            assert(currentRay.getDirection().isValid());
//
//
//            const float masPDF = its.mesh->getBSDF()->pdf(bsdfQueryRecord);
//
//            Intersection masEmitterIntersect;
//            if(!scene->rayIntersect(currentRay, masEmitterIntersect)){
//                if(t.norm() > EPSILON)
//                    return Li + t * scene->environmentEmitter.eval(currentRay);
//                else
//                    return Li;
//                //TODO handle case where bsfd sample returns zero better
//            }
//
//            if(masEmitterIntersect.mesh->isEmitter()) {
//                const float emsPDF = masEmitterIntersect.mesh->getEmitter()->pdf({currentRay.o,
//                                                                                  masEmitterIntersect.p,
//                                                                                  masEmitterIntersect.shFrame.n,
//                                                                                  masEmitterIntersect.uv});
//                wMat = masPDF + emsPDF > 0.f ? masPDF / (masPDF + emsPDF) : masPDF;
//            }
//
//            if(bsdfQueryRecord.measure == EDiscrete)
//                wMat = 1.0f;
//
//            ++numBounces;
//        }
//    }

    CPU_GPU_CONSTEXPR Color3f PathMISEnv(const Ray3f &ray, const TLAS *scene, int maxRayDepth, Sampler &sampler,
                                         FeatureBuffer *featureBuffer, size_t fbIndex) noexcept {
        Intersection its;


        Color3f Li{0.f}, t{1.f};

        Ray3f currentRay = ray;

        int numBounces = 0;

        float wMat = 1.0f;


        while(true) {
            assert(currentRay.getDirection().norm() != 0.f);
            assert(currentRay.getDirection().isValid());

            if(!scene->rayIntersect(currentRay, its)) {
                if(t.norm() > EPSILON)
                    return Li + t * scene->environmentEmitter.eval(currentRay);
                else
                    return Li;
            }

            //TMPDEBUG
//            assert(its.shFrame.n.isValid());

            //TODO fix normals for ajax

            if(numBounces == 0) {
                featureBuffer->position[fbIndex].addElement(its.p);

                const bool isDeltaDistribution = its.mesh->getBSDF().isDeltaDistribution();

                featureBuffer->normal[fbIndex].addElement(isDeltaDistribution ? Vector3f{1.f} : its.shFrame.n);

                featureBuffer->albedo[fbIndex].addElement(isDeltaDistribution ? Vector3f{1.f} : its.mesh->getBSDF()->getAlbedo(its.uv));

            }


            //environmentMap Sampling
            constexpr int maxEnvSamples = 0;
            for(int envSamples = 0; envSamples < maxEnvSamples; ++envSamples){
                EmitterQueryRecord envMapEQR{its.p};

                const Color3f envMapEMSSample = scene->environmentEmitter.sample(envMapEQR, sampler.getSample3D());

                if(!scene->rayIntersect(envMapEQR.shadowRay)) {

                    BSDFQueryRecord bsdfQueryRecord{
                            its.shFrame.toLocal(-currentRay.getDirection()),
                            its.shFrame.toLocal(envMapEQR.wi),
                            ESolidAngle};
                    bsdfQueryRecord.measure = ESolidAngle;
                    bsdfQueryRecord.uv = its.uv;

                    const float tempPDF = scene->environmentEmitter.pdf(envMapEQR);

                    Li += envMapEMSSample
                          * t
                          * its.mesh->getBSDF()->eval(bsdfQueryRecord)
                          * Frame::cosTheta(its.shFrame.toLocal(envMapEQR.wi))
                          * tempPDF
                          / (its.mesh->getBSDF()->pdf(bsdfQueryRecord) + tempPDF)
                          / maxEnvSamples
                            ;
                }
            }



            //Emitter sampling
            const auto *light = scene->getRandomEmitter(sampler.getSample1D());

            EmitterQueryRecord emitterQueryRecord{
                    its.p
            };

            const Color3f emsSample = light->sample(emitterQueryRecord, sampler.getSample3D()) * scene->numEmitters;

            if(!scene->rayIntersect(emitterQueryRecord.shadowRay)) {

                BSDFQueryRecord bsdfQueryRecord{
                        its.shFrame.toLocal(-currentRay.getDirection()),
                        its.shFrame.toLocal(emitterQueryRecord.wi),
                        ESolidAngle};
                bsdfQueryRecord.measure = ESolidAngle;
                bsdfQueryRecord.uv = its.uv;

                Li += emsSample
                      * its.mesh->getBSDF()->eval(bsdfQueryRecord)
                      * Frame::cosTheta(its.shFrame.toLocal(emitterQueryRecord.wi))
                      * light->pdf(emitterQueryRecord)
                      /(its.mesh->getBSDF()->pdf(bsdfQueryRecord) + light->pdf(emitterQueryRecord))
                      * t;
            }

            if(its.mesh->isEmitter())
                Li += t * wMat * its.mesh->getEmitter()->eval({currentRay.getOrigin(), its.p, its.shFrame.n, its.uv});

            float successProbability = fmin(t.maxCoeff(), 0.99f);
            //                if((++numBounces > 3) && sampler->next1D() > successProbability)
            if(sampler.getSample1D() >= successProbability || numBounces > maxRayDepth){
//                if(t.norm() > EPSILON)
//                    return Li + t * scene->environmentEmitter.eval(currentRay)
//                else
                    return Li;
            }

            t /= successProbability;

            BSDFQueryRecord bsdfQueryRecord{
                    its.shFrame.toLocal(-currentRay.getDirection())};
            bsdfQueryRecord.measure = ESolidAngle;
            bsdfQueryRecord.uv = its.uv;

            t *= its.mesh->getBSDF()->sample(bsdfQueryRecord, sampler.getSample2D());

            currentRay = {
                    its.p,
                    its.shFrame.toWorld(bsdfQueryRecord.wo)
            };

            const float masPDF = its.mesh->getBSDF()->pdf(bsdfQueryRecord);

            Intersection masEmitterIntersect;
            if(!scene->rayIntersect(currentRay, masEmitterIntersect)){
                if(t.norm() > EPSILON)
                    return Li + t * scene->environmentEmitter.eval(currentRay);
                else
                    return Li;
                //TODO handle case where bsfd sample returns zero better
            }

            if(masEmitterIntersect.mesh->isEmitter()) {
                const float emsPDF = masEmitterIntersect.mesh->getEmitter()->pdf({currentRay.getOrigin(),
                                                                                  masEmitterIntersect.p,
                                                                                  masEmitterIntersect.shFrame.n,
                                                                                  masEmitterIntersect.uv});
                wMat = masPDF + emsPDF > 0.f ? masPDF / (masPDF + emsPDF) : masPDF;
            }else {
                const float emsPDF = scene->environmentEmitter.pdf({
                        currentRay.getOrigin(),
                        masEmitterIntersect.p,
                        masEmitterIntersect.shFrame.n,
                        masEmitterIntersect.uv
                });

                wMat = masPDF + emsPDF > 0.f ? masPDF / (masPDF + emsPDF) : masPDF;
            }

            if(bsdfQueryRecord.measure == EDiscrete)
                wMat = 1.0f;

            ++numBounces;
        }
    }



    CPU_GPU_CONSTEXPR Color3f normalMapper(const Ray3f &ray, TLAS *scene, Sampler &sampler) noexcept {
        Intersection its;
        Color3f Li{0.f};
        if(!scene->rayIntersect(ray, its))
            return Li;


        return its.shFrame.n.absValues();
    }

    CPU_GPU_CONSTEXPR Color3f checkerboard(const Ray3f &ray, TLAS *scene, int maxRayDepth, Sampler &sampler,
                                              FeatureBuffer &featureBuffer) noexcept {
        Intersection its;

        if(!scene->rayIntersect(ray, its))
            return Color3f{0.f};

        Vector2f m_scale{0.5f, 0.5f}, m_delta{0.f, 0.f};
        Color3f m_value1{1.f}, m_value2{0.f};

        Vector2f p = its.uv / m_scale - m_delta;

        auto a = static_cast<int>(floorf(p[0]));
        auto b = static_cast<int>(floorf(p[1]));


        const int reminder = (a + b) % 2;
        const int mod = (reminder < 0) ? reminder + 2 : reminder;

        if(mod == 0.0)
            return m_value1;

        return m_value2;
    }

    CPU_GPU_CONSTEXPR Color3f depthMapper(const Ray3f &ray, TLAS *scene, Sampler &sampler) noexcept {
        Intersection its;
        Color3f Li{0.f};
        if(!scene->rayIntersect(ray, its))
            return Li;

        return (its.p + Vector3f(EPSILON)).normalized().absValues();
    }


    CPU_GPU_CONSTEXPR Color3f getColor(const Ray3f &ray, const TLAS *scene, int maxRayDepth, Sampler &sampler,
                                          FeatureBuffer *featureBuffer, size_t fbIndex) noexcept {


        //        return DirectMAS(ray, scene, sampler);
        //        return DirectMIS(ray, scene, sampler);
//                return PathMAS(ray, scene, maxRayDepth, sampler, featureBuffer);
//        return PathMIS(ray, scene, maxRayDepth, sampler, featureBuffer, fbIndex);
        return PathMISEnv(ray, scene, maxRayDepth, sampler, featureBuffer, fbIndex);
//
//        return PathVol(ray, scene, maxRayDepth, sampler, featureBuffer, fbIndex);

        //        return normalMapper(ray, scene, sampler);
        //        return depthMapper(ray, scene, sampler);
//                return checkerboard(ray, scene, maxRayDepth, sampler, featureBuffer);
    }

    __global__ void constructTLAS(TLAS *tlas,
                                  BLAS **meshBlasArr, size_t numMeshes,
                                  BLAS **emitterBlasArr, size_t numEmitters,
                                  EnvironmentEmitter environmentEmitter) ;

    template<typename T>
    [[nodiscard]] __host__ T *hostVecToDeviceRawPtr(std::vector<T> hostVec) noexcept(false) {
        T *deviceVec;
        auto numBytes = sizeof(T) * hostVec.size();

        checkCudaErrors(cudaMalloc(&deviceVec, numBytes));
        checkCudaErrors(cudaMemcpy(deviceVec, hostVec.data(), numBytes, cudaMemcpyHostToDevice));

        return deviceVec;
    }

    __global__ void applyGaussian(Vector3f *input, Vector3f *output, int width, int height, float sigma=0.1, int windowRadius=3);


    __global__ void render(Camera cam, const TLAS *tlas, int width, int height, int numSubsamples,
           int maxRayDepth, curandState *globalRandState, FeatureBuffer *featureBuffer);

    __global__ void bufferToSurface(cudaSurfaceObject_t surface, FeatureBuffer *buffer, unsigned int width, unsigned int height);
    __global__ void vecToSurface(cudaSurfaceObject_t surface, Vec3f *buffer, unsigned int width, unsigned int height);




}// namespace cudaHelpers
