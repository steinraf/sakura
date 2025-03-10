//
// Created by steinraf on 26.02.25.
//

#include "bsdf.cuh"
__host__ __device__ Color BSDF::eval(const BSDFQueryRecord &query) const noexcept {
    return material.eval(texture, query);
}
__host__ __device__ float BSDF::pdf(const BSDFQueryRecord &query) const noexcept {
    return material.pdf(query);
}
