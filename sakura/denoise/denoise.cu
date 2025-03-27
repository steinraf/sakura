//
// Created by steinraf on 15.02.25.
//

#include "../integrator/integrators.cuh"
#include "denoise.cuh"

#include "../gui/viewport.cuh"

#define denoiseGaussian denoise

__global__ void denoiseGaussian(const FeatureBuffer *bufferIn, Vec3f *output, float *weights, unsigned int width, unsigned int height) {
    for(size_t pixelIndex = blockIdx.x * blockDim.x + threadIdx.x;
        pixelIndex < width * height; pixelIndex += blockDim.x * gridDim.x) {
        unsigned int x = pixelIndex % width, y = pixelIndex / width;

        // Simply apply gaussian blur
        constexpr int MAX_RADIUS = 8;
        constexpr float color_k = 2.00f;

        Eigen::Vector3f color = Eigen::Vector3f::Zero();
        float weight = 0.0f;

        Vec3f pos = bufferIn->position[pixelIndex].getMean();

        for(int dx = -MAX_RADIUS; dx <= MAX_RADIUS; ++dx) {
            for(int dy = -MAX_RADIUS; dy <= MAX_RADIUS; ++dy) {
                int nx = safe_uint_to_int(x) + dx, ny = safe_uint_to_int(y) + dy;
                if(nx < 0 || nx >= width || ny < 0 || ny >= height) {
                    continue;
                }

                Vec3f currentColor = bufferIn->color[nx + ny * width].getMean();

                const float distanceSq = Eigen::Vector2f{dx, dy}.squaredNorm();
                const float gaussian = std::exp(-distanceSq / (color_k * color_k));

                color += currentColor * gaussian;
                weight += gaussian;
            }
        }

        Eigen::Vector3f localAverageColor = color / weight;

        output[pixelIndex] = localAverageColor;
        weights[pixelIndex] = 0.0f;
    }
}


__global__ void denoiseIdentity(const FeatureBuffer *bufferIn, Vec3f *output, float *weights, unsigned int width, unsigned int height) {

    for(size_t pixelIndex = blockIdx.x * blockDim.x + threadIdx.x;
        pixelIndex < width * height; pixelIndex += blockDim.x * gridDim.x) {
        size_t x = pixelIndex % width, y = pixelIndex / width;

        Eigen::Vector3f totalColor = bufferIn->color[pixelIndex].getMean();
        weights[pixelIndex] = 0.0f;

        output[pixelIndex] = totalColor;
        weights[pixelIndex] = 1.0f;
    }
}

#define DENOISER_EPSILON 1e-6
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