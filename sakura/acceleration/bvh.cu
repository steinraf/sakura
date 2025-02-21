//
// Created by steinraf on 04.02.25.
//

#include <utility>

#include "../geometry/intersection.cuh"
#include "../geometry/ray.cuh"
#include "../geometry/triangle.cuh"
#include "bvh.cuh"

__device__ __host__ AccelerationNode::AccelerationNode() noexcept
    ://      left(nullptr),
      //      right(nullptr),
      triangle(nullptr),
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
    : left(left),
      right(right),
      boundingBox(std::move(boundingBox)),
      isLeaf(isLeaf) {
    assert(!isLeaf);
}
__device__ __host__ AccelerationNode::AccelerationNode(Triangle *triangle,
                                                       AABB boundingBox,
                                                       bool isLeaf) noexcept
    : triangle(triangle), boundingBox(std::move(boundingBox)), isLeaf(isLeaf) {
    assert(isLeaf);
}

__device__ __host__ BVH::BVH(AccelerationNode *root,
                             size_t numTriangles) noexcept
    : root(root) {}

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
            if(currentNode->triangle->intersectHandler(currentRay, its)) {
                if(isShadowRay) {
                    return true;
                }
                closestTriangle = currentNode->triangle;
                currentRay.maxDist = its.t;
            }
            currentNode = stack[--stackIdx];// Pop
        } else {
            assert(currentNode->left != nullptr &&
                   currentNode->right != nullptr);
            AccelerationNode *left = currentNode->left;
            AccelerationNode *right = currentNode->right;

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

__device__ int findSplit(const uint32_t *mortonCodes, int first, int last,
                         int numPrimitives) {
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
            const int split_prefix =
                    delta(first, new_split, numPrimitives, mortonCodes, first_code);
            if(split_prefix > common_prefix) {
                split = new_split;// accept proposal
            }
        }
    } while(step > 1);

    return split;
}

__device__ int delta(int a, int b, unsigned int n, const unsigned int *c,
                     unsigned int ka) {
    // this guard is for leaf nodes, not internal nodes (hence [0, n-1])
    //        assert (b >= 0 && b < n);
    if(b < 0 || b > n - 1) return -1;

    unsigned int kb = c[b];
    if(ka == kb) {
        // if keys are equal, use id as fallback
        // (+32 because they have the same morton code and thus the
        // string-concatenated XOR
        //  version would have 32 leading zeros)
        return 32 + __clz(static_cast<uint32_t>(a) ^ static_cast<uint32_t>(b));
    }
    // clz = count leading zeros
    return __clz(ka ^ kb);
}

__device__ thrust::pair<int, int> determineRange(const uint32_t *mortonCodes,
                                                 int numPrimitives, int i) {
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
    const int j = i + l * d;

    //        printf("Stats of range are i=%i, j=%i, l=%i, d=%i\n", i, j, l, d);

    // ensure i <= j
    return {min(i, j), max(i, j)};
}

__global__ void constructBVH(AccelerationNode *bvhNodes, Triangle *triangles,
                             size_t numTriangles, const uint32_t *mortonCodes,
                             unsigned short *bvhNodeDone) {
    const int i = blockDim.x * blockIdx.x + threadIdx.x;

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

            int split = findSplit(mortonCodes, first, last, numTriangles);

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

        assert(&bvhNodes[childAIdx] == bvhNodes[i].left);
        assert(&bvhNodes[childBIdx] == bvhNodes[i].right);

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