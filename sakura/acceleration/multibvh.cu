//
// Created by steinraf on 25.02.25.
//

#include "../geometry/intersection.cuh"
#include "../geometry/ray.cuh"
#include "bvh.cuh"
#include "multibvh.cuh"


__device__ bool BLAS::intersect(Ray ray, Intersection &its, bool isShadowRay) const noexcept {
    ray.transform(inverseTransform);
    assert(bvh);
    //WORKS                                         return EXTREMELY_FALSE;
    bool didIntersect = bvh->intersect(ray, its, isShadowRay);
    if(didIntersect) {
        its.point = transform * its.point;
        its.shFrame.rotate(transform);
    }
    return didIntersect;
}
BLAS::BLAS(const MeshDescriptorHost &meshDescriptor) noexcept
    : bsdf(meshDescriptor.bsdf),
      bvh(nullptr),
      transform(meshDescriptor.transform),
      inverseTransform(meshDescriptor.transform.inverse()) {
    //TODO make all BVHs contiguous in memory

    bvh = getBVH(meshDescriptor.triangles);


    assert(bvh);
}
__host__ __device__ AABB BLAS::getBoundingBox() const noexcept {
    return bvh->getBoundingBox();
}

__device__ bool TLAS::intersect(const Ray &_ray, Intersection &its, bool isShadowRay) const noexcept {
    Ray ray = _ray;
    bool hit = false;
    //TODO make tree structure
    for(size_t i = 0; i < numBlas; ++i) {
        assert(i < numBlas);
        if(blas[i].intersect(ray, its, isShadowRay)) {
            if(isShadowRay) return true;
            hit = true;
            ray.maxDist = its.t;
            its.mesh = &blas[i];
        }
    }
    return hit;
}

AABB getTLASAABB(BLAS *start, size_t count) {
    AABB boundingBox = thrust::transform_reduce(
            thrust::device, start, start + count,
            [=] __host__ __device__(const BLAS &b) -> AABB {
                return b.getBoundingBox();
            },
            AABB{}, thrust::plus<AABB>());

    std::cout << "Bounding Box: " << boundingBox.min << " " << boundingBox.max << std::endl;

    return boundingBox;
}


__host__ TLAS::TLAS(const std::vector<MeshDescriptorHost> &blases) noexcept : blas(nullptr), numBlas(blases.size()) {
    assert(numBlas > 0);
    checkCudaErrors(cudaMallocManaged(&blas, numBlas * sizeof(BLAS)));
    boundingBox = {};
    for(size_t i = 0; i < numBlas; ++i) {
        blas[i] = BLAS{blases[i]};
        boundingBox = boundingBox + blas[i].getBoundingBox();
    }

    //    AABB boundingBox = getTLASAABB(blas, numBlas);
    std::cout << "Initialized TLAS with " << numBlas << " BLASes and bounding box " << boundingBox.min << " => " << boundingBox.max << '\n';


    checkCudaErrors(cudaDeviceSynchronize());
    assert(blas);
}
