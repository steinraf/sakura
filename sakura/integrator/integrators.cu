//
// Created by steinraf on 05.02.25.
//

#include "../gui/gui.cuh"
#include "integrators.cuh"

__global__ void render_kern(BVH *bvh, FeatureBuffer *buffer,
                            Camera camera, curandState *rngStates,
                            int width, int height, int spp) {


    constexpr int maxBounces = 1;

    for(size_t pixelIndex = blockIdx.x * blockDim.x + threadIdx.x;
        pixelIndex < width * height; pixelIndex += blockDim.x * gridDim.x) {
        size_t x = width - 1 - pixelIndex % width, y = pixelIndex / width;


        Sampler sampler{&rngStates[pixelIndex]};


        auto screenPos = Eigen::Vector3f{float(x) / float(width),
                                         float(y) / float(height), 0.0f};

        const Eigen::Vector3f backgroundColor =
                0.0f * Eigen::Vector3f{1.0, 1.0, 1.0};

        Eigen::Vector3f totalColor{0.0, 0.0, 0.0};
        //        const float focusDist = 77.0f;


        // cameraTf = cameraTf.inverse();
        for(int sample = 0; sample < spp; ++sample) {

            auto cameraRay =
                    camera.getRay(float(x) / width, float(y) / height, sampler);

            Intersection intersection;
            Ray currentRay = cameraRay;
            auto color = Eigen::Vector3f{0.0, 0.0, 0.0};
            auto t = Eigen::Vector3f{1.0, 1.0, 1.0};
            float etaScale = 1.0;// TODO Changes to Russian Roulette due to
                                 // indeces of refraction

            int numBounces = 0;


            while(true) {
                if(!bvh->intersect(currentRay, intersection)) {
                    //                    Eigen::Vector3f scaling =
                    //                    {1.f, 1.f, 1.f}; for (int b = 0; b <
                    //                    bounce; ++b) {
                    //                        scaling.array() *=
                    //                            Eigen::Vector3f{0.2, 0.2,
                    //                            0.2}.array();
                    //                    }
                    color.array() += t.array() * backgroundColor.array();
                    if(numBounces == 0) {
                        buffer->normal[pixelIndex].addElement(Eigen::Vector3f{0.0, 0.0, 0.0});
                        buffer->position[pixelIndex].addElement(Eigen::Vector3f{0.0, 0.0, 0.0});
                        //                        buffer->albedo[pixelIndex].addElement(Eigen::Vector3f{0.0, 0.0, 0.0});
                    }
                    break;
                } else if(numBounces == 0) {
                    buffer->normal[pixelIndex].addElement(intersection.normal);
                    buffer->position[pixelIndex].addElement(intersection.point);
                    //                    buffer->albedo[pixelIndex].addElement(intersection.material->albedo);
                }

                // Because we use roussian roulette we need to explicitly sample
                // the environment map or else almost all rays terminate
                // before reaching the end. The sun is approximated with a point
                // and randomness is applied to break the hardness of the shadow
                Eigen::Vector3f sunPos =
                        (Eigen::Vector3f{0.276, 0.2041, 0.882} +
                         sampler.getSample3D() * 0.02)
                                .normalized();

                Eigen::Vector3f envMapColor = Eigen::Vector3f{1.f, 1.f, 1.f};

                Ray shadowRay{
                        intersection.point + intersection.normal * RAY_EPSILON,
                        sunPos};
                // sample::uniformHemisphere(sampler, intersection.normal)};

                Frame intersectionFrame{intersection.normal};

                if(sampler.getSample1D() < 0.2) {
                    auto hemisphere =
                            sample::squareToCosineHemisphere(sampler.getSample2D());

                    shadowRay.dir =
                            intersectionFrame.toWorld(hemisphere).normalized();
                    envMapColor = backgroundColor;
                }

                if(!bvh->intersect(shadowRay, intersection, true)) {
                    Frame frame{intersection.normal};

                    // Grazing angles will have a lower contribution
                    float cosFactor =
                            Frame::cosTheta(frame.toLocal(shadowRay.dir));
                    color.array() +=
                            t.array() * envMapColor.array() * cosFactor;
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
        //        printf("Accessed buffer %u\n", pixelIndex);

        //        uchar4 color4 = make_uchar4(totalColor[0] * 255, totalColor[1] * 255, totalColor[2] * 255, 255);
        //        surf2Dwrite(color4, surface, x * sizeof(uchar4), y);
    }
}

__global__ void bufferToSurface(cudaSurfaceObject_t surface, FeatureBuffer *buffer, int width, int height) {
    for(size_t pixelIndex = blockIdx.x * blockDim.x + threadIdx.x;
        pixelIndex < width * height; pixelIndex += blockDim.x * gridDim.x) {
        size_t x = width - 1 - pixelIndex % width, y = pixelIndex / width;

        auto color = buffer->color[pixelIndex].getMean();

        auto clamp = [] __device__(float x, float min, float max) {
            return x < min ? min : (x > max ? max : x);
        };

        auto gammaCorrect = [&] __device__(float x) {
            if(x <= 0.0031308f) return clamp(12.92f * x, 0.f, 1.f);
            return clamp(1.055f * std::pow(x, 1.f / 2.4f) - 0.055f, 0.f,
                         1.f);
        };

        for(int c = 0; c < 3; ++c) {
            color[c] = gammaCorrect(color[c]);
            assert(color[c] <= 1.0);
        }

        uchar4 color4 = make_uchar4(color[0] * 255,
                                    color[1] * 255,
                                    color[2] * 255, 255);


        surf2Dwrite(color4, surface, x * sizeof(uchar4), y);
    }
}
