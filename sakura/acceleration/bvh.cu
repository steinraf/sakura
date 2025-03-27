//
// Created by steinraf on 04.02.25.
//

#include <utility>

#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/transform_scan.h>

#include "../geometry/intersection.cuh"
#include "../geometry/ray.cuh"
#include "../geometry/triangle.cuh"
#include "../rng/sampler.cuh"
#include "bvh.cuh"


__device__ __host__ AccelerationNode::AccelerationNode() noexcept
    ://      left(nullptr),
      //      right(nullptr),
      child(nullptr),
      boundingBox(AABB{}),
      isLeaf(false) {}

[[nodiscard]] __device__ constexpr bool AccelerationNode::hasBoundingBox()
        const noexcept {
#ifndef NDEBUG
    if(isLeaf) {
        assert(!boundingBox.isFaulty());
    }
#endif
    return isLeaf || !boundingBox.isFaulty();
}
__device__ __host__ AccelerationNode::AccelerationNode(AccelerationNode *left,
                                                       AccelerationNode *right,
                                                       AABB boundingBox,
                                                       bool isLeaf) noexcept
    : child(left, right),
      boundingBox(std::move(boundingBox)),
      isLeaf(isLeaf) {
    assert(!isLeaf);
}
__device__ __host__ AccelerationNode::AccelerationNode(Triangle *triangle,
                                                       AABB boundingBox,
                                                       bool isLeaf) noexcept
    : child(triangle), boundingBox(std::move(boundingBox)), isLeaf(isLeaf) {
    assert(isLeaf);
}

__device__ __host__ BVH::BVH(Triangle *triangles,
                             AccelerationNode *root,
                             AABB boundingBox,
                             float surfaceArea,
                             float *cdf,
                             size_t numTriangles) noexcept
    : triangleBuffer(triangles), root(root), boundingBox(std::move(boundingBox)), surfaceArea(surfaceArea), cdf(cdf), numTriangles(numTriangles) {
    assert(!this->boundingBox.isFaulty());
    assert(numTriangles != 0);
}

[[nodiscard]] __device__ bool BVH::intersect(const Ray &ray, Intersection &its,
                                             bool isShadowRay) const noexcept {
    Ray currentRay = ray;
    if(!root->boundingBox.intersect(currentRay)) {
        return false;
    }

    constexpr int stackSize = 64;
    AccelerationNode *stack[stackSize];
    int stackIdx = 0;
#ifndef NDEBUG
    int maxStackSize = 0;
#endif
    Triangle *closestTriangle = nullptr;
    AccelerationNode *currentNode = root;

    do {
        assert(stackIdx < stackSize);

        if(currentNode->isLeaf) {
            if(currentNode->getTriangle()->intersectHandler(currentRay, its)) {
                if(isShadowRay) {
                    return true;
                }
                closestTriangle = currentNode->getTriangle();
                currentRay.maxDist = its.t;
            }
            currentNode = stack[--stackIdx];// Pop
        } else {
            assert(currentNode->getLeft() != nullptr &&
                   currentNode->getRight() != nullptr);
            AccelerationNode *left = currentNode->getLeft();
            AccelerationNode *right = currentNode->getRight();

            bool continueLeft = left->boundingBox.intersect(currentRay);
            bool continueRight = right->boundingBox.intersect(currentRay);

            if(!continueLeft && !continueRight) {
                currentNode = stack[--stackIdx];// Pop
            } else {
                currentNode = continueLeft ? left : right;
                if(continueLeft) {
                    stack[stackIdx++] = right;// Push
#ifndef NDEBUG
                    if(stackIdx > maxStackSize) {
                        maxStackSize = stackIdx;
                    }
#endif
                }
            }
        }

    } while(stackIdx >= 0);

    if(closestTriangle == nullptr) return false;

    closestTriangle->hitInformationSetter(currentRay, its);

    return true;
}
__host__ __device__ AABB BVH::getBoundingBox() const noexcept {
    return boundingBox;
}
__host__ __device__ float BVH::getArea() const noexcept {
    return surfaceArea;
}
__host__ __device__ Triangle *BVH::sampleTriangle(float rng) const noexcept {
    assert(0 <= rng and rng <= 1);
    const size_t idx = sample::sampleCDF(rng, cdf, numTriangles);
    assert(idx < numTriangles);
    return triangleBuffer + idx;
}


__device__ unsigned findSplit(const uint32_t *mortonCodes, unsigned int first, unsigned int last,
                              size_t numPrimitives) {
    const unsigned int first_code = mortonCodes[first];

    // calculate the number of highest bits that are the same
    // for all objects, using the count-leading-zeros intrinsic

    const int common_prefix =
            delta(first, last, numPrimitives, mortonCodes, first_code);

    // use binary search to find where the next bit differs
    // specifically, we are looking for the highest object that
    // shares more than commonPrefix bits with the first one

    unsigned int split = first;// initial guess
    unsigned int step = last - first;

    assert(last > first);

    do {
        step = (step + 1) >> 1;                     // exponential decrease
        const unsigned int new_split = split + step;// proposed new p

        if(new_split < last) {
            const int split_prefix =
                    delta(first, new_split, numPrimitives, mortonCodes, first_code);
            if(split_prefix > common_prefix) {
                split = new_split;// accept proposal
            }
        }
    } while(step > 1);

    return split;
}

__device__ int delta(unsigned int a, unsigned int b, unsigned int n, const unsigned int *c,
                     unsigned int ka) {
    // this guard is for leaf nodes, not internal nodes (hence [0, n-1])
    if(b > n - 1) return -1;

    unsigned int kb = c[b];
    if(ka == kb) {
        // if keys are equal, use id as fallback
        // (+32 because they have the same morton code and thus the
        // string-concatenated XOR
        //  version would have 32 leading zeros)

        return 32 + __clz(safe_uint_to_int(a ^ b));
    }
    // clz = count leading zeros
    return __clz(safe_uint_to_int(ka ^ kb));
}

__device__ thrust::pair<unsigned int, unsigned int> determineRange(const uint32_t *mortonCodes,
                                                                   size_t numPrimitives, unsigned int i) {
    const unsigned int *c = mortonCodes;
    const unsigned int ki = c[i];// key of i

    // determine direction of the range (+1 or -1)
    const int delta_l = delta(i, i - 1, numPrimitives, c, ki);
    const int delta_r = delta(i, i + 1, numPrimitives, c, ki);

    const auto [delta_min, d] = [&]() -> thrust::pair<int, int> {
        if(delta_r < delta_l)
            return thrust::pair{delta_r, -1};
        else
            return thrust::pair{delta_l, 1};
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
    const unsigned int j = i + l * d;

    // ensure i <= j
    return {min(i, j), max(i, j)};
}

__global__ void constructBVH(AccelerationNode *bvhNodes, Triangle *triangles,
                             size_t numTriangles, const uint32_t *mortonCodes,
                             unsigned short *bvhNodeDone) {
    const unsigned int i = blockDim.x * blockIdx.x + threadIdx.x;

    // TODO verify if the internal nodes are one less because of root

    // [0, numPrimitives-2]                    -> internal nodes: (nullptr,
    // nullptr, Triangle *, AABB, true ) [numPrimitives-1, (2*numPrimitives)-1]
    // -> leaf nodes    : (Node *l, Node *r, nullptr,    AABB, false)

    //  Leaf Nodes are handled on the next line
    size_t childAIdx = 2 * numTriangles,
           childBIdx = 2 * numTriangles;// Invalid Index
    if(i <= numTriangles - 1) {
        // Initialize the AccelerationNode Leafs
        bvhNodes[numTriangles - 1 + i] =
                AccelerationNode{&triangles[i], triangles[i].AABBGetter()};
        assert(bvhNodes[numTriangles - 1 + i].hasBoundingBox());

        // 1 less leaf node
        if(i < numTriangles - 1) {
            auto [first, last] = determineRange(mortonCodes, numTriangles, i);

            unsigned int split = findSplit(mortonCodes, first, last, numTriangles);

            childAIdx = (split == first) ? numTriangles - 1 + split : split;
            childBIdx =
                    (split + 1 == last) ? numTriangles - 1 + split + 1 : split + 1;

            if(childAIdx > childBIdx) cuda::std::swap(childAIdx, childBIdx);

            auto *childA = &bvhNodes[childAIdx];
            auto *childB = &bvhNodes[childBIdx];

            assert(childA and childB);

            bvhNodes[i] = AccelerationNode{childA, childB, AABB{}};
            assert(!bvhNodes[i].hasBoundingBox());

            bvhNodeDone[i] = 0;
        }
    }
    __syncthreads();

    if(i < numTriangles - 1) {
        assert(childAIdx != 2 * numTriangles);
        assert(childBIdx != 2 * numTriangles);
        unsigned short int done = 1;
        unsigned short int currentValue;

        assert(&bvhNodes[childAIdx] == bvhNodes[i].getLeft());
        assert(&bvhNodes[childBIdx] == bvhNodes[i].getRight());

        // Only need to wait if childA is not a leaf
        if(childAIdx < numTriangles - 1) {
            do {
                currentValue = atomicCAS(&bvhNodeDone[childAIdx], done, done);
            } while(currentValue != done);
            assert(bvhNodes[childAIdx].hasBoundingBox());
            assert(bvhNodeDone[childAIdx] == done);
        }

        if(childBIdx < numTriangles - 1) {
            do {
                currentValue = atomicCAS(&bvhNodeDone[childBIdx], done, done);
            } while(currentValue != done);
            assert(bvhNodes[childBIdx].hasBoundingBox());
            assert(bvhNodeDone[childBIdx] == done);
        }

        bvhNodes[i].boundingBox =
                bvhNodes[childAIdx].boundingBox + bvhNodes[childBIdx].boundingBox;

        assert(bvhNodes[childAIdx].hasBoundingBox());
        assert(bvhNodes[childBIdx].hasBoundingBox());

        assert(bvhNodes[i].hasBoundingBox());

        bvhNodeDone[i] = done;
        assert(bvhNodeDone[i] == done);
        __threadfence();

        // Both Children are done computing
    }
}


BVH *getBVH(const std::vector<Triangle> &triangles) {
    Triangle *trias;
    checkCudaErrors(
            cudaMallocManaged(&trias, triangles.size() * sizeof(Triangle)));
    checkCudaErrors(cudaMemcpy(trias, triangles.data(),
                               triangles.size() * sizeof(Triangle),
                               cudaMemcpyHostToDevice));

    float totalArea = thrust::transform_reduce(
            thrust::device, trias, trias + triangles.size(),
            [=] __host__ __device__(const Triangle &t) -> float {
                return t.getArea();
            },
            0.f, thrust::plus<float>());

    AABB boundingBox = thrust::transform_reduce(
            thrust::device, trias, trias + triangles.size(),
            [=] __host__ __device__(const Triangle &t) -> AABB {
                return t.AABBGetter();
            },
            AABB{}, thrust::plus<AABB>());


    //TODO have collection of all these kinds of constants
    constexpr float EPSILON = 1e-6f;
    const Eigen::Vector3f EPS_VEC{EPSILON, EPSILON, EPSILON};

    Eigen::Vector3f lower = boundingBox.min - EPS_VEC;
    Eigen::Vector3f dims = boundingBox.max - boundingBox.min + 2 * EPS_VEC;

    thrust::device_vector<uint32_t> mortonCodes(triangles.size());
    thrust::transform(
            thrust::device, trias, trias + triangles.size(), mortonCodes.begin(),
            [=] __host__ __device__(const Triangle &tria) {
                int numBits = 10;
                const Eigen::Vector3f normalized =
                        static_cast<float>(1u << numBits) *
                        (tria.AABBGetter().getCenter() - lower).array() / dims.array();

                assert(normalized[0] >= 0 && normalized[1] >= 0 &&
                       normalized[2] >= 0);
                assert(normalized[0] <= (1u << numBits) &&
                       normalized[1] <= (1u << numBits) &&
                       normalized[2] <= (1u << numBits));

                return (LeftShift3(static_cast<uint32_t>(normalized[2])) << 2) |
                       (LeftShift3(static_cast<uint32_t>(normalized[1])) << 1) |
                       (LeftShift3(static_cast<uint32_t>(normalized[0])));
            });

    thrust::sort_by_key(thrust::device, mortonCodes.begin(), mortonCodes.end(),
                        trias);

    BVH *bvh;
    AccelerationNode *bvhNodes;

    checkCudaErrors(cudaMallocManaged(&bvh, sizeof(BVH)));
    checkCudaErrors(cudaMallocManaged(
            &bvhNodes, 2 * triangles.size() * sizeof(AccelerationNode)));

    unsigned short int *bvhConstructionDone;
    checkCudaErrors(
            cudaMallocManaged(&bvhConstructionDone,
                              (triangles.size() - 1) * sizeof(unsigned short int)));

    auto start = std::chrono::high_resolution_clock::now();
    // TODO grid stride loop or tune sizes
    constructBVH<<<(triangles.size() + 255) / 256, 256>>>(
            bvhNodes, trias, triangles.size(), mortonCodes.data().get(),
            bvhConstructionDone);

    float *cdf;
    checkCudaErrors(cudaMallocManaged(&cdf, triangles.size() * sizeof(float)));
    thrust::transform_inclusive_scan(
            trias, trias + triangles.size(), cdf,
            [=] __host__ __device__(const Triangle &t) -> float {
                return t.getArea() / totalArea;
            },
            thrust::plus<float>());

    checkCudaErrors(cudaDeviceSynchronize());
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::high_resolution_clock::now() - start);

#ifndef NDEBUG
    std::cout << "Built BVH in " << duration.count() << "ms\n";
#endif

    *bvh = BVH{
            trias,
            bvhNodes,
            boundingBox,
            totalArea,
            cdf,
            triangles.size()};

    return bvh;
}

__device__ __host__ constexpr uint32_t LeftShift3(uint32_t x) noexcept {
    if(x == (1 << 10)) --x;
    x = (x | (x << 16)) & 0b00000011000000000000000011111111;
    x = (x | (x << 8)) & 0b00000011000000001111000000001111;
    x = (x | (x << 4)) & 0b00000011000011000011000011000011;
    x = (x | (x << 2)) & 0b00001001001001001001001001001001;
    return x;
}
