//
// Created by steinraf on 23/10/22.
//

#pragma once


#include "../acceleration/aabb.h"
#include "../bsdf.h"
#include "../hittable.h"
#include "../utility/ray.h"

#define TRIANGLE_EPSILON 1e-6

struct ShapeQueryRecord {
    Vector3f ref;
    Vector3f p;
    Vector3f n;
    Vector2f uv;

    float pdf; //TODO investigate why pdf is not used

    __device__ constexpr ShapeQueryRecord() noexcept
        : ref(), p(), n(), uv(), pdf(0.f) {
    }

    __device__ constexpr ShapeQueryRecord(const Vector3f &ref_) noexcept
        : ref(ref_), p(), n(), uv(), pdf(0.f) {
    }

    __device__ constexpr ShapeQueryRecord(const Vector3f &ref_, const Vector3f &p_) noexcept
        : ref(ref_), p(p_), n(), uv(), pdf(0.f) {
    }
};


class Triangle {
public:
    constexpr Triangle() noexcept = default;

    __device__ __host__ constexpr Triangle(
            const Vector3f &p0, const Vector3f &p1, const Vector3f &p2,
            const Vector2f &uv0, const Vector2f &uv1, const Vector2f &uv2,
            const Vector3f &n0, const Vector3f &n1, const Vector3f &n2) noexcept
        : p0(p0), p1(p1), p2(p2),
          uv0(uv0), uv1(uv1), uv2(uv2),
          n0(n0), n1(n1), n2(n2){
    }

    //Nori Triangle Ray3f intersect
    __device__ constexpr bool rayIntersect(const Ray3f &r, Intersection &its) const noexcept {

        /* Find vectors for two edges sharing v[0] */
        const Vector3f edge1 = p1 - p0, edge2 = p2 - p0;

        /* Begin calculating determinant - also used to calculate U parameter */
        const Vector3f pvec = r.getDirection().cross(edge2);

        /* If determinant is near zero, ray lies in plane of triangle */
        const float det = edge1.dot(pvec);

        if(det > -TRIANGLE_EPSILON && det < TRIANGLE_EPSILON) {
            return false;
        }

        const float inv_det = 1.f / det;

        /* Calculate distance from v[0] to ray o */
        const Vector3f tvec = r.getOrigin() - p0;

        /* Calculate U parameter and test bounds */
        const float u = tvec.dot(pvec) * inv_det;
        if(u < 0.f || u > 1.f) {
            return false;
        }

        /* Prepare to test V parameter */
        const Vector3f qvec = tvec.cross(edge1);

        /* Calculate V parameter and test bounds */
        const float v = r.getDirection().dot(qvec) * inv_det;
        if(v < 0.f || u + v > 1.f) {
            return false;
        }


        const float t = edge2.dot(qvec) * inv_det;

        if(t >= r.minDist && t <= r.maxDist){
            its.uv = {u, v};
            its.t = t;

            return true;
        }

        return false;
    }

    __device__ void setHitInformation(const Ray3f &ray, Intersection &its) const noexcept;

    [[nodiscard]] __host__ __device__ constexpr inline float getArea() const noexcept {
        return 0.5f * (p1 - p0).cross(p2 - p0).norm();
    }

    [[nodiscard]] __device__ constexpr inline Vector3f getCoordinate(const Vector3f &bary) const noexcept {
        return bary[0] * p0 + bary[1] * p1 + bary[2] * p2;
    }

    [[nodiscard]] __device__ constexpr inline Vector2f getUV(const Vector3f &bary) const noexcept {
        return  bary[0] * uv0+ bary[1] * uv1+ bary[2] * uv2;;
    }

    [[nodiscard]] __device__ constexpr inline Vector3f getNormal(const Vector3f &bary) const noexcept {
        return bary[0] * n0 + bary[1] * n1 + bary[2] * n2;
    }

    [[nodiscard]] __device__ constexpr inline AABB getAABB() const noexcept {
        return AABB{p0, p1, p2};
    }

    Vector3f p0, p1, p2;
    Vector3f n0, n1, n2;
    Vector2f uv0, uv1, uv2;
};
