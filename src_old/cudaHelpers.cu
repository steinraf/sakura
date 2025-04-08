//
// Created by steinraf on 19/08/22.
//

#include "cudaHelpers.cuh"


#include "bsdf.h"
#include "utility/ray.h"

#include <thrust/pair.h>


#include <iostream>


namespace cudaHelpers {

    __device__ bool initIndices(int &i, int &j, int &pixelIndex, const int width, const int height) noexcept {
        i = threadIdx.x + blockIdx.x * blockDim.x;
        j = threadIdx.y + blockIdx.y * blockDim.y;

        if((i >= width) || (j >= height)) return false;

        pixelIndex = j * width + i;

        return true;
    }





    __global__ void initRng(int width, int height, curandState *randState) {
        int i, j, pixelIndex;
        if(!initIndices(i, j, pixelIndex, width, height)) return;

        curand_init(42, pixelIndex, 0, &randState[pixelIndex]);
    }

    __global__ void initBVH(BLAS *bvh, AccelerationNode *bvhTotalNodes, float totalArea, const float *cdf,
                            size_t numPrimitives, AreaLight *emitter, BSDF bsdf, Texture normalMap) {
        int i, j, pixelIndex;
        if(!cudaHelpers::initIndices(i, j, pixelIndex, 1, 1)) return;

        *bvh = BLAS{bvhTotalNodes, totalArea, cdf, numPrimitives, emitter, bsdf, normalMap};
    }


    enum class BOUNDARY {
        PERIODIC,
        REFLECTING,
        ZERO
    };

    __global__ void constructBVH(AccelerationNode *bvhNodes, Triangle *primitives, const uint32_t *mortonCodes, int numPrimitives) {

        const int i = blockDim.x * blockIdx.x + threadIdx.x;

        if(i > numPrimitives - 1) return;


        // [0, numPrimitives-2]                    -> internal nodes
        // [numPrimitives-1, (2*numPrimitives)-1]     -> leaf nodes
        bvhNodes[numPrimitives - 1 + i] = {
                nullptr,
                nullptr,
                &primitives[i],
                primitives[i].getAABB(),
                true,
        };


        if(i == numPrimitives - 1) return;


        auto [first, last] = determineRange(mortonCodes, numPrimitives, i);

        int split = findSplit(mortonCodes, first, last, numPrimitives);

        AccelerationNode *childA = (split == first) ? &bvhNodes[numPrimitives - 1 + split]
                                                    : &bvhNodes[split];
        AccelerationNode *childB = (split + 1 == last) ? &bvhNodes[numPrimitives - 1 + split + 1]
                                                       : &bvhNodes[split +
                                                                   1];

        bvhNodes[i] = {
                childA,
                childB,
                nullptr,
                AABB{},
                false,
        };

    }

    __device__ AABB getBoundingBox(AccelerationNode *root) noexcept {

        typedef AccelerationNode *NodePtr;

        constexpr int stackSize = 1024;
        NodePtr stack[stackSize];
        int idx = 0;
        stack[0] = root;

        assert(root);

        NodePtr currentNode;

        do {

            assert(idx < stackSize);

            currentNode = stack[idx];

            NodePtr left = currentNode->left;
            NodePtr right = currentNode->right;

            assert(left && right);

            if(left->hasBoundingBox() && right->hasBoundingBox()) {
                assert(!left->boundingBox.isEmpty() && !right->boundingBox.isEmpty());
                currentNode->boundingBox = left->boundingBox + right->boundingBox;
                --idx;
            } else if(right->hasBoundingBox()) {
                stack[++idx] = left;
            } else if(left->hasBoundingBox()) {
                stack[++idx] = right;
            } else {
                stack[++idx] = right;
                stack[++idx] = left;
            }
        } while(idx >= 0);

        return root->boundingBox;
    }

    __global__ void computeBVHBoundingBoxes(AccelerationNode *bvhNodes) {
        int i, j, pixelIndex;
        if(!cudaHelpers::initIndices(i, j, pixelIndex, 1, 1)) return;

        const AABB &totalBoundingBox = getBoundingBox(&bvhNodes[0]);

        printf("\tTotal bounding box is (%f, %f, %f) -> (%f, %f, %f)\n",
               totalBoundingBox.min[0], totalBoundingBox.min[1], totalBoundingBox.min[2],
               totalBoundingBox.max[0], totalBoundingBox.max[1], totalBoundingBox.max[2]);
    }

    __global__ void constructTLAS(TLAS *tlas,
                                  BLAS **meshBlasArr, size_t numMeshes,
                                  BLAS **emitterBlasArr, size_t numEmitters,
                                  EnvironmentEmitter environmentEmitter) {

        int i, j, pixelIndex;
        if(!cudaHelpers::initIndices(i, j, pixelIndex, 1, 1)) return;

        *tlas = TLAS(meshBlasArr, numMeshes, emitterBlasArr, numEmitters, environmentEmitter);
    }



    __global__ void applyGaussian(Vector3f *input, Vector3f *output, int width, int height, float sigma, int windowRadius){

        int i, j, pixelIndex;
        if(!initIndices(i, j, pixelIndex, width, height)) return;

        const float alpha = -1.0f / (2.0f * sigma * sigma);
        const float constant = 0.f;//std::exp(alpha * windowRadius * windowRadius);


        auto gaussian = [alpha, constant] __device__ (float xSq){
            return max(0.0f, std::exp(alpha * xSq) - constant);
        };

        float integral = 0.f;
        Vector3f tmp{0.f};

        for(int xNew = -windowRadius; xNew <= windowRadius; ++xNew) {
            if(i + xNew < 0 || i + xNew >= width) continue;

            for(int yNew = -windowRadius; yNew <= windowRadius; ++yNew){
                if(j + yNew < 0 || j + yNew >= height) continue;

                const float gaussianContrib = gaussian(xNew*xNew + yNew*yNew);
                tmp += gaussianContrib * input[(j+yNew)*width + i+xNew];
                integral += gaussianContrib;
            }
        }

        output[pixelIndex] = tmp/integral;

    }




    __device__ int findSplit(const uint32_t *mortonCodes, int first, int last, int numPrimitives) {

        const unsigned int first_code = mortonCodes[first];

        // calculate the number of highest bits that are the same
        // for all objects, using the count-leading-zeros intrinsic

        const int common_prefix =
                delta(first, last, numPrimitives, mortonCodes, first_code);

        // use binary search to find where the next bit differs
        // specifically, we are looking for the highest object that
        // shares more than commonPrefix bits with the first one

        int split = first;// initial guess
        int step = last - first;

        do {
            step = (step + 1) >> 1;            // exponential decrease
            const int new_split = split + step;// proposed new p

            if(new_split < last) {
                const int split_prefix = delta(
                        first, new_split, numPrimitives, mortonCodes, first_code);
                if(split_prefix > common_prefix) {
                    split = new_split;// accept proposal
                }
            }
        } while(step > 1);

        return split;
    }

    __device__ int delta(int a, int b, unsigned int n, const unsigned int *c, unsigned int ka) {
        // this guard is for leaf nodes, not internal nodes (hence [0, n-1])
        //        assert (b >= 0 && b < n);
        if(b < 0 || b > n - 1) return -1;

        unsigned int kb = c[b];
        if(ka == kb) {
            // if keys are equal, use id as fallback
            // (+32 because they have the same morton code and thus the string-concatenated XOR
            //  version would have 32 leading zeros)
            return 32 + __clz(static_cast<uint32_t>(a) ^ static_cast<uint32_t>(b));
        }
        // clz = count leading zeros
        return __clz(ka ^ kb);
    }

    __device__ std::pair<int, int> determineRange(const uint32_t *mortonCodes, int numPrimitives, int i){
        const unsigned int *c = mortonCodes;
        const unsigned int ki = c[i];// key of i

        // determine direction of the range (+1 or -1)
        const int delta_l = delta(i, i - 1, numPrimitives, c, ki);
        const int delta_r = delta(i, i + 1, numPrimitives, c, ki);

        const auto [d, delta_min] = [&]() -> const thrust::pair<int, int> {
            if(delta_r < delta_l)
                return thrust::pair{-1, delta_r};
            else
                return thrust::pair{1, delta_l};
        }();

        // compute upper bound of the length of the range
        unsigned int l_max = 2;
        while(delta(i, i + l_max * d, numPrimitives, c, ki) > delta_min) {
            l_max <<= 1;
        }

        // find other end using binary search
        unsigned int l = 0;
        for(unsigned int t = l_max >> 1; t > 0; t >>= 1) {
            if(delta(i, i + (l + t) * d, numPrimitives, c, ki) > delta_min) {
                l += t;
            }
        }
        const int j = i + l * d;

        //        printf("Stats of range are i=%i, j=%i, l=%i, d=%i\n", i, j, l, d);

        // ensure i <= j
        return std::make_pair(std::min(i, j), std::max(i, j));
    }


    __global__ void render(Camera cam, TLAS *tlas, int width, int height, int numSubsamples,
                           int maxRayDepth, curandState *globalRandState, FeatureBuffer *featureBuffer) {
        int i, j, pixelIndex;
        if(!initIndices(i, j, pixelIndex, width, height)) return;


        auto sampler = Sampler(&globalRandState[pixelIndex]);

        const auto iFloat = static_cast<float>(i);
        const auto jFloat = static_cast<float>(j);

        const auto widthFloat = static_cast<float>(width);
        const auto heightFloat = static_cast<float>(height);

        int actualSamples = numSubsamples;

        for(int subSamples = 0; subSamples < numSubsamples; ++subSamples) {

            const float s = (iFloat + sampler.getSample1D()) / (widthFloat + 1);
            const float t = (jFloat + sampler.getSample1D()) / (heightFloat + 1);

            const auto ray = cam.getRay(s, t, sampler.getSample2D());

            const Vector3f currentColor = getColor(ray, tlas, maxRayDepth, sampler, featureBuffer, pixelIndex);

            featureBuffer->color[pixelIndex].addElement(currentColor);
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

    __global__ void bufferToSurface(cudaSurfaceObject_t surface, FeatureBuffer *buffer, unsigned int width, unsigned int height) {
        for(size_t pixelIndex = blockIdx.x * blockDim.x + threadIdx.x;
            pixelIndex < width * height; pixelIndex += blockDim.x * gridDim.x) {
            size_t x = pixelIndex % width, y = pixelIndex / width;

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

    __global__ void vecToSurface(cudaSurfaceObject_t surface, Vec3f *buffer, unsigned int width, unsigned int height) {
        for(size_t pixelIndex = blockIdx.x * blockDim.x + threadIdx.x;
            pixelIndex < width * height; pixelIndex += blockDim.x * gridDim.x) {
            size_t x = pixelIndex % width, y = pixelIndex / width;

            auto color = tonemap(buffer[pixelIndex]);

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

}// namespace cudaHelpers
