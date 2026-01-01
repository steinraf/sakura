//
// Created by steinraf on 04.02.25.
//

#include <utility>

#include "../acceleration/aabb.cuh"
#include "../geometry/intersection.cuh"
#include "../geometry/ray.cuh"
#include "triangle.cuh"

CPU_GPU Triangle::Triangle(
        Vec3f p0, Vec3f p1, Vec3f p2,
        const Vec3f &n0, const Vec3f &n1, const Vec3f &n2,
        Vec2f uv0, Vec2f uv1, Vec2f uv2) noexcept
    : p0(std::move(p0)),
      p1(std::move(p1)),
      p2(std::move(p2)),
      n0(n0.normalized()),
      n1(n1.normalized()),
      n2(n2.normalized()),
      uv0(std::move(uv0)),
      uv1(std::move(uv1)),
      uv2(std::move(uv2)) {}

__device__ Triangle::Triangle() noexcept
    : p0{1, 0, 0},
      p1{0, 1, 0},
      p2{0, 0, 1},
      n0{1, 0, 0},
      n1{0, 1, 0},
      n2{0, 0, 1},
      uv0{0.0, 0.0},
      uv1{1.0, 0.0},
      uv2{0.0, 1.0} {}

CPU_GPU bool Triangle::intersectHandler(
        const Ray &ray, Intersection &its) const noexcept {
    // Nori ray intersection code

    /* Find vectors for two edges sharing v[0] */
    const Vec3f edge1 = p1 - p0, edge2 = p2 - p0;

    // TODO check if triangle is degenerate
    // https://github.com/mmp/pbrt-v4/blob/39e01e61f8de07b99859df04b271a02a53d9aeb2/src/pbrt/shapes.cpp#L175

    /* Begin calculating determinant - also used to calculate U parameter */
    const Vec3f pvec = ray.dir.cross(edge2);

    /* If determinant is near zero, ray lies in plane of triangle */
    const float det = edge1.dot(pvec);

    if(det > -TRIANGLE_COLLISION_EPSILON && det < TRIANGLE_COLLISION_EPSILON) {
        return false;
    }

    const float inv_det = 1.f / det;

    /* Calculate distance from v[0] to ray o */
    const Vec3f tvec = ray.origin - p0;

    /* Calculate U parameter and test bounds */
    const float u = tvec.dot(pvec) * inv_det;
    if(u < 0.f || u > 1.f) {
        return false;
    }

    /* Prepare to test V parameter */
    const Vec3f qvec = tvec.cross(edge1);

    /* Calculate V parameter and test bounds */
    const float v = ray.dir.dot(qvec) * inv_det;
    if(v < 0.f || u + v > 1.f) {
        return false;
    }

    const float t = edge2.dot(qvec) * inv_det;

    if(t >= ray.minDist && t <= ray.maxDist) {
        its.uv = {u, v};
        its.t = t;

        return true;
    }

    return false;
}

CPU_GPU AABB Triangle::AABBGetter() const noexcept {
    return {p0, p1, p2};
}

CPU_GPU void Triangle::hitInformationSetter(
        const Ray &r, Intersection &its) const noexcept {
    float u = its.uv[0];
    float v = its.uv[1];

    const Vec3f bary = {1.0f - u - v, u, v};

    its.point = getCoordinate(bary);
    its.uv = getUV(bary);
    Vec3f normal = getNormal(bary);

    //Flip normal if it is facing away from the ray
    if(normal.dot(r.dir) > 0) {
        its.shFrame = Frame(-normal);
    } else {
        its.shFrame = Frame(normal);
    }

    its.triangle = this;
}
CPU_GPU float Triangle::getArea() const noexcept {
    return 0.5f * (p1 - p0).cross(p2 - p0).norm();
}
CPU_GPU Vec3f Triangle::getCoordinate(const Vec3f &c) const noexcept {
    return c[0] * p0 + c[1] * p1 + c[2] * p2;
}
CPU_GPU Vec3f Triangle::getNormal(const Vec3f &c) const noexcept {
    return c[0] * n0 + c[1] * n1 + c[2] * n2;
}
CPU_GPU Vec2f Triangle::getUV(const Vec3f &c) const noexcept {
    return c[0] * uv0 + c[1] * uv1 + c[2] * uv2;
}
