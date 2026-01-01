//
// Created by steinraf on 15.02.25.
//

#include "../integrator/integrators.cuh"
#include "denoise.cuh"

#include "../gui/viewport.cuh"

#define denoiseGaussian denoise


#define DENOISER_EPSILON 1e-6

__global__ void denoiseGaussian(const FeatureBuffer *bufferIn, Vec3f *output, float *weights, unsigned int width, unsigned int height) {
    for(size_t pixelIndex = blockIdx.x * blockDim.x + threadIdx.x;
        pixelIndex < width * height; pixelIndex += blockDim.x * gridDim.x) {
        unsigned int x = pixelIndex % width, y = pixelIndex / width;

        // Simply apply gaussian blur
        constexpr int MAX_RADIUS = 4;
        constexpr float color_k = 5.20f, spatial_k = 0.02f, var_k = 1.0f;

        Vec3f color = Vec3f::Zero();
        float weight = 0.0f;

        Vec3f pos = bufferIn->position[pixelIndex].getMean();
        Vec3f var = bufferIn->color[pixelIndex].getSampleVariance();
        Vec3f normal = bufferIn->normal[pixelIndex].getMean();

        for(int dx = -MAX_RADIUS; dx <= MAX_RADIUS; ++dx) {
            for(int dy = -MAX_RADIUS; dy <= MAX_RADIUS; ++dy) {
                int nx = safe_uint_to_int(x) + dx, ny = safe_uint_to_int(y) + dy;
                if(nx < 0 || nx >= width || ny < 0 || ny >= height) {
                    continue;
                }

                Vec3f currentColor = bufferIn->color[nx + ny * width].getMean();
                Vec3f currentVar = bufferIn->color[nx + ny * width].getSampleVariance();
                Vec3f currentNormal = bufferIn->normal[nx + ny * width].getMean();

                const float distanceSq = Vec2f{dx, dy}.squaredNorm();
                const float spatialDistSq = (bufferIn->position[nx + ny * width].getMean() - pos).squaredNorm();
                const float normalDot = std::abs(normal.dot(currentNormal));

                const float varRatio = [&]() -> float {
                    if(currentVar.minCoeff() < DENOISER_EPSILON) {
                        return var.maxCoeff() / DENOISER_EPSILON;
                    } else {
                        return (var.array() / currentVar.array()).maxCoeff();
                    }
                }();

                const float gaussianScreenSpace = std::exp(-distanceSq / (color_k * color_k));
                const float gaussianSpatial = std::exp(-spatialDistSq / (spatial_k * spatial_k));
                const float gaussianVar = std::exp(-varRatio / (var_k * var_k));

                float w = gaussianSpatial * normalDot;

                color += currentColor * w;
                weight += w;
            }
        }

        if(weight < DENOISER_EPSILON) weight = DENOISER_EPSILON;

        Vec3f localAverageColor = color / weight;

        output[pixelIndex] = localAverageColor;
        weights[pixelIndex] = 0.0f;
    }
}


__global__ void denoiseIdentity(const FeatureBuffer *bufferIn, Vec3f *output, float *weights, unsigned int width, unsigned int height) {

    for(size_t pixelIndex = blockIdx.x * blockDim.x + threadIdx.x;
        pixelIndex < width * height; pixelIndex += blockDim.x * gridDim.x) {
        size_t x = pixelIndex % width, y = pixelIndex / width;

        Vec3f totalColor = bufferIn->color[pixelIndex].getMean();
        weights[pixelIndex] = 0.0f;

        output[pixelIndex] = totalColor;
        weights[pixelIndex] = 1.0f;
    }
}


__global__ void applyWeights(cudaSurfaceObject_t surface, const Vec3f *buffer, float *weights, unsigned int width, unsigned int height) {


    for(size_t pixelIndex = blockIdx.x * blockDim.x + threadIdx.x;
        pixelIndex < width * height; pixelIndex += blockDim.x * gridDim.x) {
        size_t x = width - 1 - pixelIndex % width, y = pixelIndex / width;

        float weight = weights[pixelIndex];

        auto color = tonemap([&]() -> Color {
            auto c = buffer[pixelIndex];
            if(weight < DENOISER_EPSILON) {
                return c;
            }
            return c / weight;
        }());

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