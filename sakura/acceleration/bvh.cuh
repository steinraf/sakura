//
// Created by steinraf on 04.02.25.
//

#pragma once

#include "../acceleration/aabb.cuh"
#include "../common.cuh"

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
    union Child {
        struct {
            Triangle *triangle;
        };
        struct {
            AccelerationNode *left;
            AccelerationNode *right;
        };
        CPU_GPU explicit Child(Triangle *triangle) : triangle(triangle) {}
        CPU_GPU explicit Child(AccelerationNode *left, AccelerationNode *right)
            : left(left), right(right) {}
    } child;

    AABB boundingBox;
    bool isLeaf;

    __device__ __host__ Triangle *getTriangle() const {
        assert(isLeaf);
        return child.triangle;
    }

    __device__ __host__ inline AccelerationNode *getLeft() const {
        assert(!isLeaf);
        return child.left;
    }

    __device__ __host__ inline AccelerationNode *getRight() const {
        assert(!isLeaf);
        return child.right;
    }
};

// Bounding Volume Hierarchy
class BVH {
private:
    Triangle *triangleBuffer;
    AccelerationNode *root;
    AABB boundingBox;

    float surfaceArea;

    float *cdf;
    size_t numTriangles;


public:
    __device__ __host__ explicit BVH(Triangle *triangles,
                                     AccelerationNode *root,
                                     AABB boundingBox,
                                     float surfaceArea /* surface of geometry, not related to SAH */,
                                     float *cdf,
                                     size_t numTriangles) noexcept;


    [[nodiscard]] __device__ bool intersect(
            const Ray &ray, Intersection &its,
            bool isShadowRay = false) const noexcept;

    [[nodiscard]] CPU_GPU AABB getBoundingBox() const noexcept;

    [[nodiscard]] CPU_GPU float getArea() const noexcept;

    [[nodiscard]] CPU_GPU Triangle *sampleTriangle(float rng) const noexcept;
    bool intersectAABB(Ray ray, Intersection &intersection);
};

// The findSplit, delta and determineRange are taken from here
// https://developer.nvidia.com/blog/thinking-parallel-part-iii-tree-construction-gpu/
// https://github.com/nolmoonen/cuda-lbvh/blob/main/src/build.cu
__device__ unsigned findSplit(const uint32_t *mortonCodes, unsigned int first, unsigned int last,
                              size_t numPrimitives);

__device__ int delta(unsigned int a, unsigned int b, unsigned int n, const unsigned int *c,
                     unsigned int ka);

__device__ thrust::pair<unsigned int, unsigned int> determineRange(const uint32_t *mortonCodes,
                                                                   size_t numPrimitives, unsigned int i);

__global__ void constructBVH(AccelerationNode *bvhNodes, Triangle *triangles,
                             size_t numTriangles, const uint32_t *mortonCodes,
                             unsigned short int *bvhNodeDone);


BVH *getBVH(const std::vector<Triangle> &triangles);

__device__ __host__ constexpr uint32_t LeftShift3(uint32_t x) noexcept;