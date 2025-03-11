//
// Created by steinraf on 25.02.25.
//

#pragma once

#include "../common.cuh"
#include "../geometry/triangle.cuh"
#include "../material/bsdf.cuh"
#include "../rng/sampler.cuh"
#include "aabb.cuh"

struct MeshDescriptorHost {
    std::vector<Triangle> triangles;
    Eigen::Affine3f transform;
    BSDF bsdf;
};

class BLAS {
public:
    __host__ explicit BLAS(const MeshDescriptorHost &meshDescriptor) noexcept;

    //    __host__ __device__ BLAS(const BLAS &other) noexcept = delete;
    //    __host__ __device__ BLAS &operator=(const BLAS &other) noexcept = delete;
    //    __host__ __device__ BLAS(BLAS &&other) noexcept = delete;
    //    __host__ __device__ BLAS &operator=(BLAS &&other) noexcept = default;

    [[nodiscard]] __host__ __device__ AABB getBoundingBox() const noexcept;// todo account for transform

    [[nodiscard]] __device__ bool intersect(
            Ray ray, Intersection &its,
            bool isShadowRay = false) const noexcept;

    BSDF bsdf;

private:
    BVH *bvh;
    Eigen::Affine3f inverseTransform;
    friend TLAS;
};


class TLAS {
public:
    __host__ explicit TLAS(const std::vector<MeshDescriptorHost> &blases) noexcept;


    [[nodiscard]] __device__ bool intersect(
            const Ray &ray, Intersection &its,
            bool isShadowRay = false) const noexcept;

    void __host__ cleanup() {
        if(blas) return;

        checkCudaErrors(cudaFree(blas));
    }

    __device__ void shuffleTfs(Sampler &sampler) {
        for(size_t i = 0; i < numBlas; i++) {

            Eigen::Affine3f tf = blas[i].inverseTransform.inverse();
            //            tf.translation() += (float(i) - numBlas / 2) * Vec3f::Ones() / numBlas * 0.1f;
            auto rotation = Eigen::AngleAxisf(sampler.getSample1D() * 2 * M_PI, Vec3f::UnitX());
            //            tf.rotate(rotation);
            blas[i].inverseTransform = tf.inverse();
            //            break;
        }
    }

    __host__ __device__ AABB getBoundingBox() const {
        return boundingBox;
    }


private:
    BLAS *blas;
    size_t numBlas;
    AABB boundingBox;// todo account for transform
};