//
// Created by steinraf on 04.02.25.
//

#include <utility>

#include "../acceleration/aabb.cuh"
#include "../geometry/intersection.cuh"
#include "../geometry/ray.cuh"
#include "triangle.cuh"

__host__ __device__ Triangle::Triangle(
        Eigen::Vector3f p0, Eigen::Vector3f p1, Eigen::Vector3f p2,
        const Eigen::Vector3f &n0, const Eigen::Vector3f &n1,
        const Eigen::Vector3f &n2
        //             const Eigen::Vector2f& uv0, const Eigen::Vector2f&
        //             uv1, const Eigen::Vector2f& uv2
        ) noexcept
    : p0(std::move(p0)),
      p1(std::move(p1)),
      p2(std::move(p2)),
      n0(n0.normalized()),
      n1(n1.normalized()),
      n2(n2.normalized())
//          uv0(uv0),
//          uv1(uv1),
//          uv2(uv2)
{}

__device__ Triangle::Triangle() noexcept
    : p0{1, 0, 0},
      p1{0, 1, 0},
      p2{0, 0, 1},
      n0{1, 0, 0},
      n1{0, 1, 0},
      n2{0, 0, 1}//          uv0{0.0, 0.0},
                 //          uv1{1.0, 0.0},
                 //          uv2{0.0, 1.0}
{}

__host__ __device__ bool Triangle::intersectHandler(
        const Ray &ray, Intersection &its) const noexcept {
    // Nori ray intersection code

    /* Find vectors for two edges sharing v[0] */
    const Eigen::Vector3f edge1 = p1 - p0, edge2 = p2 - p0;

    // TODO check if triangle is degenerate
    // https://github.com/mmp/pbrt-v4/blob/39e01e61f8de07b99859df04b271a02a53d9aeb2/src/pbrt/shapes.cpp#L175

    /* Begin calculating determinant - also used to calculate U parameter */
    const Eigen::Vector3f pvec = ray.dir.cross(edge2);

    /* If determinant is near zero, ray lies in plane of triangle */
    const float det = edge1.dot(pvec);

    if(det > -TRIANGLE_COLLISION_EPSILON && det < TRIANGLE_COLLISION_EPSILON) {
        return false;
    }

    const float inv_det = 1.f / det;

    /* Calculate distance from v[0] to ray o */
    const Eigen::Vector3f tvec = ray.origin - p0;

    /* Calculate U parameter and test bounds */
    const float u = tvec.dot(pvec) * inv_det;
    if(u < 0.f || u > 1.f) {
        return false;
    }

    /* Prepare to test V parameter */
    const Eigen::Vector3f qvec = tvec.cross(edge1);

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

__host__ __device__ AABB Triangle::AABBGetter() const noexcept {
    return {p0, p1, p2};
}

__host__ __device__ void Triangle::hitInformationSetter(
        const Ray &r, Intersection &its) const noexcept {
    float u = its.uv[0];
    float v = its.uv[1];

    const Eigen::Vector3f bary = {1.0f - u - v, u, v};

    its.point = Eigen::Vector3f{bary[0] * p0 + bary[1] * p1 + bary[2] * p2};

    its.uv = {u, v};


    Vec3f normal = Eigen::Vector3f{bary[0] * n0 + bary[1] * n1 + bary[2] * n2};

    //Flip normal if it is facing away from the ray
    if(normal.dot(r.dir) > 0) {
        its.shFrame = Frame(-normal);
    } else {
        its.shFrame = Frame(normal);
    }

    its.triangle = this;
}