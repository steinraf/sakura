//
// Created by steinraf on 19/08/22.
//


#include "scene.h"

#include <fstream>

#include <OpenImageDenoise/oidn.hpp>

#include <cuda_gl_interop.h>
#include <thrust/transform_reduce.h>


__host__ Scene::Scene(SceneRepresentation &&sceneRepr)
    : deviceCamera(sceneRepr.cameraInfo.origin,
          sceneRepr.cameraInfo.target,
          sceneRepr.cameraInfo.up,
          sceneRepr.cameraInfo.fov,
          static_cast<float>(sceneRepr.sceneInfo.width) / static_cast<float>(sceneRepr.sceneInfo.height),
          sceneRepr.cameraInfo.aperture,
          sceneRepr.cameraInfo.focusDist,
          sceneRepr.cameraInfo.k1,
          sceneRepr.cameraInfo.k2,
          sceneRepr.cameraInfo.apertureType),
     sceneRepresentation(sceneRepr),
     hostDeviceMeshTriangleVec(sceneRepresentation.meshInfos.size()),
     hostDeviceMeshCDF(sceneRepresentation.meshInfos.size()),
     totalMeshArea(sceneRepresentation.meshInfos.size()),
     hostDeviceEmitterTriangleVec(sceneRepresentation.emitterInfos.size()),
     hostDeviceEmitterCDF(sceneRepresentation.emitterInfos.size()),
     totalEmitterArea(sceneRepresentation.emitterInfos.size()){

    checkCudaErrors(cudaMalloc(&meshAccelerationStructure, sizeof(TLAS)));

    auto numMeshes = sceneRepresentation.meshInfos.size();

    std::vector<BLAS *> hostMeshBlasVector(numMeshes);

    clock_t meshLoadStart = clock();
#pragma omp parallel for
    for(size_t i = 0; i < numMeshes; ++i) {
        hostMeshBlasVector[i] = getMeshFromFile(sceneRepr.meshInfos[i].filename,
                                                hostDeviceMeshTriangleVec[i],
                                                hostDeviceMeshCDF[i],
                                                totalMeshArea[i],
                                                sceneRepr.meshInfos[i].transform,
                                                sceneRepr.meshInfos[i].bsdf,
                                                sceneRepr.meshInfos[i].normalMap,
                                                nullptr);
    }

    auto numEmitters = sceneRepresentation.emitterInfos.size();

    if(numEmitters == 0){
        std::cerr << "The scene seems to not contain any emitters. Please add a valid one and try again.\n";
        exit(1);
    }

    std::vector<BLAS *> hostEmitterBlasVector(numEmitters);

    std::vector<AreaLight> hostAreaLights(numEmitters);
    for(size_t i = 0; i < numEmitters; ++i) {
        hostAreaLights[i] = AreaLight(sceneRepr.emitterInfos[i].radiance);
    }

    AreaLight *deviceAreaLights;
    checkCudaErrors(cudaMalloc(&deviceAreaLights, sizeof(AreaLight) * numEmitters));
    checkCudaErrors(cudaMemcpy(deviceAreaLights, hostAreaLights.data(), sizeof(AreaLight) * numEmitters,
                               cudaMemcpyHostToDevice));


#pragma omp parallel for
    for(size_t i = 0; i < numEmitters; ++i) {
        hostEmitterBlasVector[i] = getMeshFromFile(sceneRepr.emitterInfos[i].filename,
                                                   hostDeviceEmitterTriangleVec[i],
                                                   hostDeviceEmitterCDF[i],
                                                   totalEmitterArea[i],
                                                   sceneRepr.emitterInfos[i].transform,
                                                   sceneRepr.emitterInfos[i].bsdf,
                                                   sceneRepr.emitterInfos[i].normalMap,
                                                   deviceAreaLights + i);
    }


    std::cout << "Loading all Geometry took "
              << ((double) (clock() - meshLoadStart)) / CLOCKS_PER_SEC
              << " seconds.\n";


    BLAS **deviceBlasArr = cudaHelpers::hostVecToDeviceRawPtr(hostMeshBlasVector);
    BLAS **deviceEmitterBlasArr = cudaHelpers::hostVecToDeviceRawPtr(hostEmitterBlasVector);

    cudaHelpers::constructTLAS<<<1, 1>>>(meshAccelerationStructure,
                                         deviceBlasArr, numMeshes,
                                         deviceEmitterBlasArr, numEmitters,
                                         EnvironmentEmitter{sceneRepresentation.environmentInfo.texture});

    checkCudaErrors(cudaGetLastError());


}


TLAS const * Scene::getMeshAccelerationStructure() {
    return meshAccelerationStructure;
}
