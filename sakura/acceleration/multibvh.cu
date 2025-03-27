//
// Created by steinraf on 25.02.25.
//

#include "../emitter/emitter.cuh"
#include "../geometry/intersection.cuh"
#include "../geometry/ray.cuh"
#include "bvh.cuh"
#include "multibvh.cuh"

__device__ bool BLAS::intersect(Ray ray, Intersection &its, bool isShadowRay) const noexcept {
    ray.transform(inverseTransform);
    assert(bvh);
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
__host__ __device__ float BLAS::pdfSurface(const ShapeQueryRecord &sqr) const noexcept {

    return 1.0f / (bvh->getArea() * transform.linear().determinant());
}
__host__ __device__ void BLAS::sampleSurface(ShapeQueryRecord &sqr, const Vec3f &rng) const noexcept {
    const Vec3f p = sample::squareToUniformTriangle({rng[0], rng[1]});
    const auto triangle = bvh->sampleTriangle(rng[2]);
    assert(triangle);


    sqr.normal = transform.linear() * triangle->getNormal(p);
    sqr.point = transform * triangle->getCoordinate(p);
    sqr.uv = triangle->getUV(p);

    sqr.pdf = pdfSurface(sqr);// TODO check if this should be pdf of triangle?
}


__device__ bool TLAS::intersect(const Ray &_ray, Intersection &its, bool isShadowRay) const noexcept {
    Ray ray = _ray;
    bool hit = false;
    //TODO make tree structure
    for(size_t i = 0; i < numMeshes; ++i) {
        if(meshes[i].intersect(ray, its, isShadowRay)) {
            if(isShadowRay) return true;
            hit = true;
            ray.maxDist = its.t;
            its.meshf = &meshes[i];
            its.emitter = nullptr;
        }
    }
    for(size_t i = 0; i < numEmitters; ++i) {
        if(emitters[i].intersect(ray, its, isShadowRay)) {
            if(isShadowRay) return true;
            hit = true;
            ray.maxDist = its.t;
            its.meshf = emitters[i].blas;
            its.emitter = &emitters[i];
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


__host__ TLAS::TLAS(const std::vector<MeshDescriptorHost> &_meshes, const std::vector<EmitterDescriptorHost> &_emitters) noexcept
    : meshes(nullptr), numMeshes(_meshes.size()),
      emitters(nullptr), numEmitters(_emitters.size()) {

    checkCudaErrors(cudaMallocManaged(&meshes, numMeshes * sizeof(BLAS)));
    checkCudaErrors(cudaMallocManaged(&emitters, numEmitters * sizeof(AreaLight)));
    boundingBox = {};

    for(size_t i = 0; i < numMeshes; ++i) {
        meshes[i] = BLAS{_meshes[i]};
        boundingBox = boundingBox + meshes[i].getBoundingBox();
    }

    for(size_t i = 0; i < numEmitters; ++i) {
        emitters[i] = AreaLight{_emitters[i]};
    }

    std::cout << "Initialized TLAS with " << numMeshes << " meshes and " << numEmitters << " emitters and bounding box " << boundingBox.min << " => " << boundingBox.max << '\n';

    checkCudaErrors(cudaDeviceSynchronize());
    assert(meshes);
}
__host__ __device__ const AreaLight *TLAS::getRandomEmitter(float d) {
    if(numEmitters == 0) return nullptr;

    if(d == 1.0f) d = 0.0f;//curand_uniform random numbers are in (0, 1]
    size_t idx = std::floor(d * numEmitters);
    assert(idx < numEmitters);
    return &emitters[idx];
}
__host__ void __host__ TLAS::cleanup() {
    if(!meshes) return;
    if(!emitters) return;

    checkCudaErrors(cudaFree(meshes));
    checkCudaErrors(cudaFree(emitters));
}
__device__ bool TLAS::intersect(const Ray &ray) const noexcept {
    Intersection its;
    return intersect(ray, its, true);
}
