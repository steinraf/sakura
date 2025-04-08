//
// Created by steinraf on 07.04.25.
//

#include "featureBuffer.cuh"

__host__ FeatureBuffer::FeatureBuffer(size_t numElements) : numElements(numElements), color(nullptr), normal(nullptr), position(nullptr), albedo(nullptr), uv(nullptr) {
    checkCudaErrors(cudaMallocManaged(&color, numElements * sizeof(Statistic<Vec3f>)));
    checkCudaErrors(cudaMallocManaged(&normal, numElements * sizeof(Statistic<Vec3f>)));
    checkCudaErrors(cudaMallocManaged(&position, numElements * sizeof(Statistic<Vec3f>)));
    checkCudaErrors(cudaMallocManaged(&albedo, numElements * sizeof(Statistic<Vec3f>)));
    checkCudaErrors(cudaMallocManaged(&uv, numElements * sizeof(Statistic<Vec3f>)));
    clear();
}

__host__ FeatureBuffer::~FeatureBuffer() {
    checkCudaErrors(cudaFree(color));
    checkCudaErrors(cudaFree(normal));
    checkCudaErrors(cudaFree(position));
    checkCudaErrors(cudaFree(albedo));
    checkCudaErrors(cudaFree(uv));
}
void FeatureBuffer::clear() {
    int threadsPerBlock = 256;
    size_t blocksPerGrid = (numElements + threadsPerBlock - 1) / threadsPerBlock;

    clearFeatureBuffer<<<blocksPerGrid, threadsPerBlock>>>(this);
    checkCudaErrors(cudaDeviceSynchronize());
}

__global__ void decayFeatureBuffer(FeatureBuffer *buffer, float k) {

    auto idx = blockIdx.x * blockDim.x + threadIdx.x;

    if(idx >= buffer->numElements) {
        return;
    }

    buffer->color[idx].scale(k);
    buffer->normal[idx].scale(k);
    buffer->position[idx].scale(k);
    buffer->albedo[idx].scale(k);
    buffer->uv[idx].scale(k);
}

void FeatureBuffer::decay(float k) {
    int threadsPerBlock = 256;
    size_t blocksPerGrid = (numElements + threadsPerBlock - 1) / threadsPerBlock;

    decayFeatureBuffer<<<blocksPerGrid, threadsPerBlock>>>(this, k);
    checkCudaErrors(cudaDeviceSynchronize());
}



__global__ void clearFeatureBuffer(FeatureBuffer *buffer) {

    auto idx = blockIdx.x * blockDim.x + threadIdx.x;

    if(idx >= buffer->numElements) {
        return;
    }

    buffer->color[idx] = {};
    buffer->normal[idx] = {};
    buffer->position[idx] = {};
    buffer->albedo[idx] = {};
    buffer->uv[idx] = {};
}

