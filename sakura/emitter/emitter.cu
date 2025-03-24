//
// Created by steinraf on 13.03.25.
//

#include "../geometry/ray.cuh"
#include "emitter.cuh"

AreaLight::AreaLight(const EmitterDescriptorHost &emitterDescriptor) noexcept
    : blas([&emitterDescriptor]() {
          BLAS *blas;
          checkCudaErrors(cudaMallocManaged(&blas, sizeof(BLAS)));
          MeshDescriptorHost meshDescriptor{
                  emitterDescriptor.triangles,
                  emitterDescriptor.transform,
                  emitterDescriptor.bsdf};
          *blas = BLAS(meshDescriptor);
          return blas;
      }()),
      radiance(emitterDescriptor.radiance) {
}
__device__ bool AreaLight::intersect(const Ray &ray, Intersection &its, bool isShadowRay) const noexcept {
    return blas->intersect(ray, its, isShadowRay);
}
__device__ AABB AreaLight::getBoundingBox() const noexcept {
    return blas->getBoundingBox();
}
__device__ Color AreaLight::sample(EmitterQueryRecord &eqr, Vec3f rng) const {
    assert(false);
    return Color::Zero();
}
