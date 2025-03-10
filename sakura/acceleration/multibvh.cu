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
    return bvh->intersect(ray, its, isShadowRay);
}
BLAS::BLAS(const std::vector<Triangle> &triangles, const Eigen::Affine3f &transform) noexcept : bvh(nullptr), inverseTransform(transform.inverse()) {
    //TODO make all BVHs contiguous in memory


    //    checkCudaErrors(cudaMallocManaged(&bvh, sizeof(BVH)));

    bvh = getBVH(triangles);

    std::cout << "Allocated BVH at " << bvh << std::endl;
    assert(bvh);
}

BLAS::~BLAS() {
    std::cout << "Starting BLAS destruct\n";
    //    if(bvh) {
    //        std::cout << "Freeing BVH " << bvh << "\n";
    //        checkCudaErrors(cudaFree(bvh));
    //    }
    std::cout << "Destructed BLAS\n";
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
        }
    }
    return hit;
}
__host__ TLAS::TLAS(const std::vector<MeshDescriptorHost> &blases) noexcept : blas(nullptr), numBlas(blases.size()) {
    assert(numBlas > 0);
    checkCudaErrors(cudaMallocManaged(&blas, numBlas * sizeof(BLAS)));
    for(size_t i = 0; i < numBlas; ++i) {
        std::cout << "Initializing BLAS " << i << " at " << blas + i << std::endl;
        blas[i] = BLAS{blases[i].triangles, blases[i].transform};
        std::cout << "Initialized BLAS " << i << " at " << blas + i << std::endl;
    }
    std::cout << "Initialized TLAS with " << numBlas << " BLASes at " << blas << std::endl;
    checkCudaErrors(cudaDeviceSynchronize());
    assert(blas);
}
