//
// Created by steinraf on 04.02.25.
//

#pragma once

#include "../geometry/triangle.cuh"
#include "aabb.cuh"

// Acceleration Node is a binary tree node for use in the BVH
// Leaf Nodes:
//     triangle = Triangle *t
//     boundingBox = t->getAABB()
//     isLeaf = true
// Internal Nodes:
//     left = AccelerationNode *l, right = AccelerationNode *r
//     boundingBox = l->boundingBox + r->boundingBox
//     isLeaf = false
struct AccelerationNode {
    __device__ __host__ AccelerationNode() noexcept;

    __device__ __host__ AccelerationNode(AccelerationNode *left,
                                         AccelerationNode *right,
                                         AABB boundingBox,
                                         bool isLeaf = false) noexcept;

    __device__ __host__ AccelerationNode(Triangle *triangle, AABB boundingBox,
                                         bool isLeaf = true) noexcept;

    [[nodiscard]] __device__ constexpr bool hasBoundingBox() const noexcept;

    // Either has a triangle or two children nodes
    union {
        struct {
            Triangle *triangle;
        };
        struct {
            AccelerationNode *left;
            AccelerationNode *right;
        };
    };
    AABB boundingBox;
    bool isLeaf;
};

// Bounding Volume Hierarchy
class BVH {
private:
    AccelerationNode *root;

public:
    __device__ __host__ explicit BVH(AccelerationNode *root,
                                     size_t numTriangles) noexcept;

    [[nodiscard]] __device__ bool intersect(
            const Ray &ray, Intersection &its,
            bool isShadowRay = false) const noexcept;
};

// The findSplit, delta and determineRange are taken from here
// https://developer.nvidia.com/blog/thinking-parallel-part-iii-tree-construction-gpu/
// https://github.com/nolmoonen/cuda-lbvh/blob/main/src/build.cu
__device__ int findSplit(const uint32_t *mortonCodes, int first, int last,
                         int numPrimitives);

__device__ int delta(int a, int b, unsigned int n, const unsigned int *c,
                     unsigned int ka);

__device__ thrust::pair<int, int> determineRange(const uint32_t *mortonCodes,
                                                 int numPrimitives, int i);

__global__ void constructBVH(AccelerationNode *bvhNodes, Triangle *triangles,
                             size_t numTriangles, const uint32_t *mortonCodes,
                             unsigned short int *bvhNodeDone);