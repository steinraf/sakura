//
// Created by steinraf on 02/12/22.
//

#include "../cudaHelpers.cuh"
#include "imageTexture.h"
#include <thrust/device_vector.h>
#include <thrust/transform_scan.h>
#include <thrust/transform_reduce.h>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"


__host__ Texture::Texture(const std::filesystem::path &imagePath, bool isEnvMap) noexcept {
    assert(!imagePath.string().empty());
    width = height = dim = 0;
    printf("Loading texture %s\n", imagePath.c_str());
    auto *hostTexture = (Vector3f *) stbi_loadf(imagePath.c_str(), &width, &height, &dim, 3);

#ifndef NDEBUG
    if(!hostTexture){
        printf("The failure reason is %s\n", stbi__g_failure_reason);
        assert(false);
    }
#endif

    assert(hostTexture);


    printf("Size of the image is %i / %i\n", width, height);


    checkCudaErrors(cudaMalloc(&deviceTexture, width * height * sizeof(Vector3f)));
    checkCudaErrors(cudaMalloc(&deviceCDF, width * height * sizeof(float)));

    checkCudaErrors(cudaMemcpy(deviceTexture, hostTexture, width * height * sizeof(Vector3f), cudaMemcpyHostToDevice));


    ColorToRadiance colorToRadiance(deviceTexture, width, height, isEnvMap);


    thrust::device_ptr<Vector3f> deviceTexturePtr{deviceTexture};
    float totalSum = thrust::transform_reduce(deviceTexturePtr, deviceTexturePtr + width*height,
                                                    colorToRadiance, 0.f, thrust::plus<float>());

    ColorToCDF colorToCdf{deviceTexture, width, height, totalSum, isEnvMap};

    thrust::transform_inclusive_scan(deviceTexturePtr, deviceTexturePtr + width*height,
                                     deviceCDF, colorToCdf, thrust::plus<float>());

    assert(dim == 3);
}