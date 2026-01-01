//
// Created by steinraf on 13.03.25.
//

#include <utility>

#include "../geometry/ray.cuh"
#include "emitter.cuh"

constexpr float EMITTER_EPSILON = 1e-6f;

AreaLight::AreaLight(const EmitterDescriptorHost &emitterDescriptor) noexcept
    : blas([&emitterDescriptor]() {
          BLAS *blas;
          checkCudaErrors(cudaMallocManaged(&blas, sizeof(BLAS)));

          MeshDescriptorHost meshDescriptor{
                  emitterDescriptor.triangles,
                  emitterDescriptor.transform,
                  [&]() {
                      if(emitterDescriptor.bsdf.hasZeroTexture()) {
                          printf("WARNING: Arealight has zero texture. This would lead to no radiance. changing to constant texture.\n");
                          auto bsdf = emitterDescriptor.bsdf;
                          bsdf.setUnitTexture();
                          return bsdf;
                      } else {
                          return emitterDescriptor.bsdf;
                      }
                  }()};
          *blas = BLAS(meshDescriptor);
          return blas;
      }()),
      radiance(emitterDescriptor.radiance) {
    assert(blas);
}
__device__ bool AreaLight::intersect(const Ray &ray, Intersection &its, bool isShadowRay) const noexcept {
    return blas->intersect(ray, its, isShadowRay);
}
__device__ AABB AreaLight::getBoundingBox() const noexcept {
    return blas->getBoundingBox();
}
__device__ Color AreaLight::sample(EmitterQueryRecord &eqr, const Vec3f &rng) const {
    ShapeQueryRecord sqr{eqr.ref};

    blas->sampleSurface(sqr, rng);

    eqr.point = sqr.point;
    eqr.wIn = (eqr.point - eqr.ref).normalized();
    eqr.shadowRay = Ray{
            eqr.ref,
            eqr.wIn,
            RAY_EPSILON,
            (eqr.point - eqr.ref).norm() - RAY_EPSILON};

    eqr.normal = sqr.normal.normalized();
    eqr.uv = sqr.uv;
    eqr.pdf = pdf(eqr);

    return eval(eqr) / eqr.pdf;
}
__device__ float AreaLight::pdf(const EmitterQueryRecord &eqr) const noexcept {
    ShapeQueryRecord sqr{
            eqr.ref,
            eqr.point,
    };

    return (eqr.ref - eqr.point).squaredNorm() * blas->pdfSurface(sqr) / abs(eqr.normal.dot(-eqr.wIn) + EMITTER_EPSILON);
}
__device__ Color AreaLight::eval(const EmitterQueryRecord &eqr) const noexcept {
    if(eqr.normal.dot(eqr.wIn) >= 0.0f) {
        return Color::Zero();
    }
    return radiance.array() * blas->bsdf.evalTexture(eqr.uv).array();
}
__device__ bool AreaLight::intersectAABB(Ray ray, Intersection &its) const {
    return blas->intersectAABB(std::move(ray), its);
}
