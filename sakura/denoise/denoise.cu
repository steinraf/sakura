//
// Created by steinraf on 15.02.25.
//

#include "../gui/gui.cuh"
#include "../integrator/integrators.cuh"
#include "denoise.cuh"

#include <Eigen/Dense>

__global__ void denoiseOld(cudaSurfaceObject_t surface, const FeatureBuffer *buffer, int width, int height) {
    for(size_t pixelIndex = blockIdx.x * blockDim.x + threadIdx.x;
        pixelIndex < width * height; pixelIndex += blockDim.x * gridDim.x) {
        int x = width - 1 - pixelIndex % width, y = pixelIndex / width;

        constexpr int MAX_RADIUS = 8;

        Eigen::Vector3f localAverageColor{0, 0, 0};
        int avgCount = 0;
        for(int dx = -MAX_RADIUS; dx < MAX_RADIUS + 1; ++dx) {
            for(int dy = -MAX_RADIUS; dy < MAX_RADIUS + 1; ++dy) {

                if(x + dx < 0 || x + dx >= width || y + dy < 0 || y + dy >= height) continue;
                localAverageColor += buffer->color[(y + dy) * width + (width - 1 - (x + dx))].getMean();
                ++avgCount;
            }
        }
        localAverageColor /= avgCount;

        Eigen::Vector3f totalColor = (0.4f * buffer->color[pixelIndex].getMean() + 0.6f * localAverageColor);

        totalColor = tonemap(totalColor);

        uchar4 color4 = make_uchar4(totalColor[0] * 255, totalColor[1] * 255, totalColor[2] * 255, 255);
        surf2Dwrite(color4, surface, x * sizeof(uchar4), y);
    }
}

// Bilateral filter

__device__ Eigen::Vector3f bilateralFilter(const FeatureBuffer *buffer, int width, int height, int x, int y) {


    return Eigen::Vector3f{0, 0, 0};
}

__global__ void denoise(cudaSurfaceObject_t surface, const FeatureBuffer *buffer, int width, int height) {

    for(size_t pixelIndex = blockIdx.x * blockDim.x + threadIdx.x;
        pixelIndex < width * height; pixelIndex += blockDim.x * gridDim.x) {
        int x = width - 1 - pixelIndex % width, y = pixelIndex / width;

        Eigen::Vector3f color = bilateralFilter(buffer, width, height, width - 1 - x, y);

        color = tonemap(color);

        uchar4 color4 = make_uchar4(color[0] * 255, color[1] * 255, color[2] * 255, 255);
        surf2Dwrite(color4, surface, x * sizeof(uchar4), y);
    }
}