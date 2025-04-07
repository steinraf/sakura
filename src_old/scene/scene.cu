//
// Created by steinraf on 19/08/22.
//


#include "scene.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#include <fstream>
#include <thread>

#include <OpenImageDenoise/oidn.hpp>

#include <thrust/transform_reduce.h>


__host__ Scene::Scene(SceneRepresentation &&sceneRepr, Device dev) : sceneRepresentation(sceneRepr),
                                                                     imageBufferByteSize(sceneRepr.sceneInfo.width * sceneRepr.sceneInfo.height * sizeof(Vector3f)),
                                                                     blockSize(sceneRepr.sceneInfo.width / blockSizeX + 1, sceneRepr.sceneInfo.height / blockSizeY + 1),
                                                                     device(dev),
                                                                     hostDeviceMeshTriangleVec(sceneRepresentation.meshInfos.size()),
                                                                     hostDeviceMeshCDF(sceneRepresentation.meshInfos.size()),
                                                                     totalMeshArea(sceneRepresentation.meshInfos.size()),
                                                                     hostDeviceEmitterTriangleVec(sceneRepresentation.emitterInfos.size()),
                                                                     hostDeviceEmitterCDF(sceneRepresentation.emitterInfos.size()),
                                                                     totalEmitterArea(sceneRepresentation.emitterInfos.size()),
                                                                     deviceCamera(sceneRepr.cameraInfo.origin,
                                                                                  sceneRepr.cameraInfo.target,
                                                                                  sceneRepr.cameraInfo.up,
                                                                                  sceneRepr.cameraInfo.fov,
                                                                                  static_cast<float>(sceneRepr.sceneInfo.width) / static_cast<float>(sceneRepr.sceneInfo.height),
                                                                                  sceneRepr.cameraInfo.aperture,
                                                                                  sceneRepr.cameraInfo.focusDist,
                                                                                  sceneRepr.cameraInfo.k1,
                                                                                  sceneRepr.cameraInfo.k2,
                                                                                  sceneRepr.cameraInfo.apertureType) {

    assert(dev == CPU);

    const auto numPixels = sceneRepr.sceneInfo.width * sceneRepr.sceneInfo.height;
    if(dev == CPU) {
        checkCudaErrors(cudaMalloc(&deviceImageBuffer, imageBufferByteSize));
        checkCudaErrors(cudaMemset(deviceImageBuffer, 0.f, imageBufferByteSize));
        checkCudaErrors(cudaMalloc(&deviceImageBufferDenoised, imageBufferByteSize));
    } else {
        //GPU mode does not work properly yet.
        //TODO fix GPU drawing mode
        //        checkCudaErrors(cudaMalloc(&deviceImageBuffer, imageBufferByteSize)); Allocated by OpenGL instead
        checkCudaErrors(cudaMalloc(&deviceImageBufferDenoised, imageBufferByteSize));
    }

    checkCudaErrors(cudaMalloc(&deviceFeatureBuffer.variances, imageBufferByteSize));
    checkCudaErrors(cudaMalloc(&deviceFeatureBuffer.albedos  , imageBufferByteSize));
    checkCudaErrors(cudaMalloc(&deviceFeatureBuffer.positions, imageBufferByteSize));
    checkCudaErrors(cudaMalloc(&deviceFeatureBuffer.normals  , imageBufferByteSize));

    checkCudaErrors(cudaMalloc(&deviceFeatureBuffer.numSubSamples, sizeof(int) * numPixels));


    checkCudaErrors(cudaMemset(deviceFeatureBuffer.variances, 0.f, imageBufferByteSize));
    checkCudaErrors(cudaMemset(deviceFeatureBuffer.albedos  , 0.f, imageBufferByteSize));
    checkCudaErrors(cudaMemset(deviceFeatureBuffer.positions, 0.f, imageBufferByteSize));
    checkCudaErrors(cudaMemset(deviceFeatureBuffer.normals  , 0.f, imageBufferByteSize));

    checkCudaErrors(cudaMemset(deviceFeatureBuffer.numSubSamples, 0.f, sizeof(int) * numPixels));



    checkCudaErrors(cudaMalloc(&deviceCurandState, sizeof(curandState) * numPixels));

    cudaHelpers::initRng<<<blockSize, threadSize>>>(sceneRepresentation.sceneInfo.width, sceneRepresentation.sceneInfo.height, deviceCurandState);
    checkCudaErrors(cudaGetLastError());


    checkCudaErrors(cudaMalloc(&meshAccelerationStructure, sizeof(TLAS)));

    // No need to sync because can run independently
    //    checkCudaErrors(cudaDeviceSynchronize());

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


    hostImageBuffer = new Vector3f[imageBufferByteSize];
    hostImageBufferDenoised = new Vector3f[imageBufferByteSize];

}

__host__ Scene::~Scene() {

    //TODO properly cuda free buffers

    delete[] hostImageBuffer;
    delete[] hostImageBufferDenoised;

    if(device == CPU) {
        checkCudaErrors(cudaDeviceSynchronize());
    } else {

    }
    cudaHelpers::freeVariables<<<blockSize, threadSize>>>();
}

void Scene::reset() noexcept{
    checkCudaErrors(cudaMemset(deviceImageBuffer, 0.f, imageBufferByteSize));
    checkCudaErrors(cudaMemset(deviceFeatureBuffer.variances, 0.f, imageBufferByteSize));
    checkCudaErrors(cudaMemset(deviceFeatureBuffer.albedos  , 0.f, imageBufferByteSize));
    checkCudaErrors(cudaMemset(deviceFeatureBuffer.positions, 0.f, imageBufferByteSize));
    checkCudaErrors(cudaMemset(deviceFeatureBuffer.normals  , 0.f, imageBufferByteSize));
    checkCudaErrors(cudaMemset(deviceFeatureBuffer.numSubSamples, 0.f, sizeof(int) * sceneRepresentation.sceneInfo.width * sceneRepresentation.sceneInfo.height));

    actualSamples = 0;
}


bool Scene::render() {

    //TODO fix GPU drawing mode

    volatile bool currentlyRendering = true;

//    std::thread drawingThread;

//    if(device == GPU) {
//        drawingThread = std::thread{[this](Vector3f *v, volatile bool &render) {
//                                        OpenGLDraw(v, render);
//                                    },
//                                    deviceImageBuffer, std::ref(currentlyRendering)};
//    }


//    std::cout << "Starting render...\n";

//    std::cout << "Starting Rendering...";

    clock_t startRender = clock();


    if(actualSamples >= sceneRepresentation.sceneInfo.samplePerPixel)
        return false;

    //Exponentially increasing number of samples
    const auto samplesRemaining = CustomRenderer::clamp(1, 1, sceneRepresentation.sceneInfo.samplePerPixel - actualSamples);
//    auto samplesRemaining = CustomRenderer::min(CustomRenderer::max(1, actualSamples), );

    cudaHelpers::render<<<blockSize, threadSize>>>(deviceImageBuffer, deviceCamera, meshAccelerationStructure,
                                                   sceneRepresentation.sceneInfo.width, sceneRepresentation.sceneInfo.height, samplesRemaining, sceneRepresentation.sceneInfo.maxRayDepth,
                                                   deviceCurandState, deviceFeatureBuffer, nullptr);
    checkCudaErrors(cudaGetLastError());


    checkCudaErrors(cudaDeviceSynchronize());


    actualSamples += samplesRemaining;

    checkCudaErrors(cudaMemcpy(hostImageBuffer, deviceImageBuffer, imageBufferByteSize, cudaMemcpyDeviceToHost));

//    std::cout << "\rRendering took " << ((double) (clock() - startRender)) / CLOCKS_PER_SEC << " seconds.\n";


#ifndef NDEBUG

    ColorToNorm colorToNorm;

    thrust::device_ptr<Vector3f> deviceTexturePtr{deviceImageBuffer};
    float totalSum = thrust::transform_reduce(deviceTexturePtr, deviceTexturePtr + sceneRepresentation.sceneInfo.width * sceneRepresentation.sceneInfo.height,
                                              colorToNorm, 0.f, thrust::plus<float>());
#endif

    if(actualSamples == samplesRemaining){

    }

    if(!hostImageTexture)
        glGenTextures(1, &hostImageTexture);

    glBindTexture(GL_TEXTURE_2D, hostImageTexture);

    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

#if defined(GL_UNPACK_ROW_LENGTH) && !defined(__EMSCRIPTEN__)
    glPixelStorei(GL_UNPACK_ROW_LENGTH, 0);
#endif
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, sceneRepresentation.sceneInfo.width, sceneRepresentation.sceneInfo.height, 0, GL_RGB, GL_FLOAT, hostImageBuffer);


    return true;

}

__host__ void Scene::denoise() {

    clock_t startDenoise = clock();

    std::cout << "Starting denoise...";

    float *deviceWeights;

    checkCudaErrors(cudaMalloc(&deviceWeights, sceneRepresentation.sceneInfo.width * sceneRepresentation.sceneInfo.height * sizeof(float)));
    checkCudaErrors(cudaMemset(deviceWeights, 0.f, sceneRepresentation.sceneInfo.width * sceneRepresentation.sceneInfo.height * sizeof(float)));

    Vector3f *varianceCopy;
    checkCudaErrors(cudaMalloc(&varianceCopy, imageBufferByteSize));
    checkCudaErrors(cudaMemcpy(varianceCopy, deviceFeatureBuffer.variances, imageBufferByteSize, cudaMemcpyDeviceToDevice));
    cudaHelpers::applyGaussian<<<blockSize, threadSize>>>(varianceCopy, deviceFeatureBuffer.variances, sceneRepresentation.sceneInfo.width, sceneRepresentation.sceneInfo.height, 0.3f, 21);

    checkCudaErrors(cudaDeviceSynchronize());
    checkCudaErrors(cudaFree(varianceCopy));


//        denoise<<<blockSize, threadSize>>>(deviceImageBuffer, deviceImageBufferDenoised, deviceFeatureBuffer, deviceWeights, sceneRepresentation.sceneInfo.width, sceneRepresentation.sceneInfo.height, sceneRepresentation.cameraInfo.origin);
//        checkCudaErrors(cudaDeviceSynchronize());
    //    denoiseApplyWeights<<<blockSize, threadSize>>>(deviceImageBufferDenoised, deviceWeights, sceneRepresentation.sceneInfo.width, sceneRepresentation.sceneInfo.height);
    //    checkCudaErrors(cudaDeviceSynchronize());
    //    checkCudaErrors(cudaMemcpy(hostImageBufferDenoised, deviceImageBufferDenoised,
    //                               imageBufferByteSize, cudaMemcpyDeviceToHost));

    checkCudaErrors(cudaFree(deviceWeights));





    auto oidnDevice = oidn::newDevice();
    oidnDevice.commit();

    auto numDev = oidn::getNumPhysicalDevices();
    if(numDev == 0){
        std::cerr << "No OIDN devices found.\n";
    }

    const char* errorMessage;
    if (oidnDevice.getError(errorMessage) != oidn::Error::None){
        std::cerr << "OIDN Error: " << errorMessage << '\n';
        return;
    }

    int width = sceneRepresentation.sceneInfo.width;
    int height = sceneRepresentation.sceneInfo.height;

    std::vector<Vector3f> albedos(width*height);

    checkCudaErrors(cudaMemcpy(albedos.data(), deviceFeatureBuffer.albedos, imageBufferByteSize, cudaMemcpyDeviceToHost));

    std::vector<Vector3f> normals(width*height);
    checkCudaErrors(cudaMemcpy(normals.data(), deviceFeatureBuffer.normals, imageBufferByteSize, cudaMemcpyDeviceToHost));

    checkCudaErrors(cudaDeviceSynchronize());


    oidn::FilterRef filter = oidnDevice.newFilter("RT");
    filter.setImage("color",  hostImageBuffer,  oidn::Format::Float3, width, height);
    filter.setImage("albedo", albedos.data(), oidn::Format::Float3, width, height);
    filter.setImage("normal", normals.data(), oidn::Format::Float3, width, height);
    filter.setImage("output", hostImageBufferDenoised, oidn::Format::Float3, width, height);
    filter.commit();

    filter.execute();

    if (oidnDevice.getError(errorMessage) != oidn::Error::None)
        std::cerr << "Error: " << errorMessage << '\n';


    std::cout << "\rDenoising took " << ((double) (clock() - startDenoise)) / CLOCKS_PER_SEC << " seconds.\n";

}


__host__ void Scene::saveOutput() {

    std::cout << "Writing resulting image to disk...\n";

    if(!std::filesystem::exists("./data"))
        std::filesystem::create_directory("./data");

    const std::string pngPath = "./data/image.png";
    const std::string pngPathDenoised = "./data/imageDenoised.png";

    const std::string hdrPath = "./data/image.hdr";
    const std::string hdrPathDenoised = "./data/imageDenoised.hdr";

    const bool didHDR = stbi_write_hdr(hdrPath.c_str(), sceneRepresentation.sceneInfo.width,
                                       sceneRepresentation.sceneInfo.height, 3, (float *) hostImageBuffer);
    assert(didHDR);

    const bool didHDRDenoised = stbi_write_hdr(hdrPathDenoised.c_str(), sceneRepresentation.sceneInfo.width,
                                       sceneRepresentation.sceneInfo.height, 3, (float *) hostImageBufferDenoised);
    assert(didHDRDenoised);


    pngwriter png(sceneRepresentation.sceneInfo.width, sceneRepresentation.sceneInfo.height, 1., pngPath.c_str());
    pngwriter pngDenoised(sceneRepresentation.sceneInfo.width, sceneRepresentation.sceneInfo.height, 1.,
                          pngPathDenoised.c_str());


#pragma omp parallel for
    for(int j = 0; j < sceneRepresentation.sceneInfo.height; j++) {
        for(int i = 0; i < sceneRepresentation.sceneInfo.width; i++) {
            const int idx = j * sceneRepresentation.sceneInfo.width + i;
            png.plot(i + 1, sceneRepresentation.sceneInfo.height - j,
                     Warp::gammaCorrect(hostImageBuffer[idx][0]),
                     Warp::gammaCorrect(hostImageBuffer[idx][1]),
                     Warp::gammaCorrect(hostImageBuffer[idx][2]));
            pngDenoised.plot(i + 1, sceneRepresentation.sceneInfo.height - j,
                             Warp::gammaCorrect(hostImageBufferDenoised[idx][0]),
                             Warp::gammaCorrect(hostImageBufferDenoised[idx][1]),
                             Warp::gammaCorrect(hostImageBufferDenoised[idx][2]));
        }
    }

    png.close();
    pngDenoised.close();

    std::cout << "Saving images completed.\n";
}

__host__ void Scene::step(float dt) noexcept {
    deviceCamera.addVelocityRelative(cameraVelocity, dt);
    std::cout << std::flush;
}
