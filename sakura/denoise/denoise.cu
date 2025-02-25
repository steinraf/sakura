//
// Created by steinraf on 15.02.25.
//

#include "../integrator/integrators.cuh"
#include "denoise.cuh"

#include "../gui/viewport.cuh"

__global__ void denoise(cudaSurfaceObject_t surface, const FeatureBuffer *buffer, unsigned int width, unsigned int height) {
    for(size_t pixelIndex = blockIdx.x * blockDim.x + threadIdx.x;
        pixelIndex < width * height; pixelIndex += blockDim.x * gridDim.x) {
        unsigned int x = width - 1 - pixelIndex % width, y = pixelIndex / width;

        constexpr int MAX_RADIUS = 32;

        // Simply apply gaussian blur

        Eigen::Vector3f color = Eigen::Vector3f::Zero();
        float weight = 0.0f;

        for(int dx = -MAX_RADIUS; dx <= MAX_RADIUS; ++dx) {
            for(int dy = -MAX_RADIUS; dy <= MAX_RADIUS; ++dy) {
                int nx = safe_uint_to_int(x) + dx, ny = safe_uint_to_int(y) + dy;
                if(nx < 0 || nx >= width || ny < 0 || ny >= height) {
                    continue;
                }

                Eigen::Vector3f currentColor = buffer->color[width - 1 - nx + ny * width].getMean();

                const float distanceSq = Eigen::Vector2f{dx, dy}.squaredNorm();
                const float gaussian = std::exp(-distanceSq / (2 * 2));

                color += currentColor * gaussian;
                weight += gaussian;
            }
        }

        Eigen::Vector3f localAverageColor = color / weight;

        Eigen::Vector3f totalColor = (localAverageColor);

        totalColor = tonemap(totalColor);

        auto toChar = [](float x) {
            return static_cast<unsigned char>(std::clamp(x * 255.f, 0.f, 255.f));
        };

        uchar4 color4 = make_uchar4(toChar(totalColor[0]),
                                    toChar(totalColor[1]),
                                    toChar(totalColor[2]),
                                    255);
        surf2Dwrite(color4, surface, x * sizeof(uchar4), y);
    }
}