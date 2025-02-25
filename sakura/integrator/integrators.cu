//
// Created by steinraf on 05.02.25.
//

#include "../acceleration/bvh.cuh"
#include "../camera/camera.cuh"
#include "../geometry/intersection.cuh"
#include "../gui/viewport.cuh"
#include "../rng/sampler.cuh"
#include "integrators.cuh"

__global__ void render_kern(BVH *bvh, FeatureBuffer *buffer,
                            Camera camera, curandState *rngStates,
                            unsigned int width, unsigned int height, int spp) {


    constexpr int maxBounces = 5;

    for(size_t pixelIndex = blockIdx.x * blockDim.x + threadIdx.x;
        pixelIndex < width * height; pixelIndex += blockDim.x * gridDim.x) {
        size_t x = pixelIndex % width, y = pixelIndex / width;


        Sampler sampler{&rngStates[pixelIndex]};


        auto screenPos = Eigen::Vector3f{float(x) / float(width),
                                         float(y) / float(height), 0.0f};

        const Eigen::Vector3f backgroundColor =
                1.0f * Eigen::Vector3f{1.0, 1.0, 1.0};

        Eigen::Vector3f totalColor{0.0, 0.0, 0.0};

        for(int sample = 0; sample < spp; ++sample) {

            auto cameraRay =
                    camera.getRay(float(x) / float(width), float(y) / float(height), sampler);


            Intersection intersection;
            Ray currentRay = cameraRay;
            auto color = Eigen::Vector3f{0.0, 0.0, 0.0};
            auto t = Eigen::Vector3f{1.0, 1.0, 1.0};
            float etaScale = 1.0;// TODO Changes to Russian Roulette due to
                                 // indices of refraction

            int numBounces = 0;

            while(true) {
                if(!bvh->intersect(currentRay, intersection)) {

                    color.array() += t.array() * backgroundColor.array();
                    if(numBounces == 0) {
                        buffer->normal[pixelIndex].addElement(Eigen::Vector3f{0.0, 0.0, 0.0});
                        buffer->position[pixelIndex].addElement(Eigen::Vector3f{0.0, 0.0, 0.0});
                        //                        buffer->albedo[pixelIndex].addElement(Eigen::Vector3f{0.0, 0.0, 0.0});
                        buffer->uv[pixelIndex].addElement(Eigen::Vector3f{0.0, 0.0, 0.0});
                    }
                    break;
                } else if(numBounces == 0) {
                    buffer->normal[pixelIndex].addElement(intersection.normal);
                    buffer->position[pixelIndex].addElement(intersection.point);
                    //                    buffer->albedo[pixelIndex].addElement(intersection.material->albedo);
                    buffer->uv[pixelIndex].addElement(Eigen::Vector3f{intersection.uv[0], intersection.uv[1], 0.0});
                }


                // roulette
                float successProbability = min(t.maxCoeff() * etaScale, 0.99f);
                if(sampler.getSample1D() > successProbability ||
                   numBounces > maxBounces) {
                    break;
                }

                t.array() /= successProbability;

                t.array() *= 0.8;// BSDF

                currentRay = Ray{
                        intersection.point + intersection.normal * RAY_EPSILON,
                        sample::uniformHemisphere(sampler, intersection.normal)};

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


__device__ Eigen::Vector3f tonemap(Eigen::Vector3f color) {

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