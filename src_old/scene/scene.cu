//
// Created by steinraf on 19/08/22.
//


#include "scene.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "../../src/denoise/denoise.h"
#include "stb_image_write.h"

#include <fstream>

#include <OpenImageDenoise/oidn.hpp>

#include <cuda_gl_interop.h>
#include <thrust/transform_reduce.h>


__host__ Scene::Scene(SceneRepresentation &&sceneRepr) : sceneRepresentation(sceneRepr),
                                                                     blockSize(sceneRepr.sceneInfo.width / blockSizeX + 1, sceneRepr.sceneInfo.height / blockSizeY + 1),
                                                                     imageBuffer(static_cast<size_t>(sceneRepr.sceneInfo.width), static_cast<size_t>(sceneRepr.sceneInfo.height)),
                                                                     imageBufferDenoised(static_cast<size_t>(sceneRepr.sceneInfo.width), static_cast<size_t>(sceneRepr.sceneInfo.height)),
                                                                     denoiseWeights(nullptr),
                                                                     denoiseOutput(nullptr),
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
                                                                                  sceneRepr.cameraInfo.apertureType)
                                                                    {


    checkCudaErrors(cudaMallocManaged(&denoiseWeights, sizeof(float) * sceneRepr.sceneInfo.width * sceneRepr.sceneInfo.height));
    checkCudaErrors(cudaMallocManaged(&denoiseOutput, sizeof(Vec3f) * sceneRepr.sceneInfo.width * sceneRepr.sceneInfo.height));

    const auto numPixels = sceneRepr.sceneInfo.width * sceneRepr.sceneInfo.height;


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


}


void Scene::reset() noexcept{

    //todo decay
    imageBuffer.featureBuffer->clear();
    imageBufferDenoised.featureBuffer->clear();

//    imageBuffer.featureBuffer->decay(0.5f);
//    imageBufferDenoised.featureBuffer->decay(0.5f);


    actualSamples = 0;
}


bool Scene::render() {

    //TODO fix GPU drawing mode

    if(actualSamples >= sceneRepresentation.sceneInfo.samplePerPixel)
        return false;

    int spp = 1;

    int devId = 0;
    int numSMs;
    checkCudaErrors(cudaDeviceGetAttribute(&numSMs, cudaDevAttrMultiProcessorCount, devId));
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    assert(imageBuffer.featureBuffer);

    cudaHelpers::render<<<blockSize, threadSize>>>(deviceCamera, meshAccelerationStructure,
                                                   sceneRepresentation.sceneInfo.width, sceneRepresentation.sceneInfo.height, spp, sceneRepresentation.sceneInfo.maxRayDepth,
                                                   deviceCurandState, imageBuffer.featureBuffer);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());






    cudaHelpers::bufferToSurface<<<32 * numSMs, 256>>>(imageBuffer.surface, imageBuffer.featureBuffer, sceneRepresentation.sceneInfo.width, sceneRepresentation.sceneInfo.height);
    checkCudaErrors(cudaDeviceSynchronize());

    if(denoiserEnabled){
        denoiser<<<blockSize, threadSize>>>(imageBuffer.featureBuffer, denoiseOutput, denoiseWeights, sceneRepresentation.sceneInfo.width, sceneRepresentation.sceneInfo.height);
        checkCudaErrors(cudaDeviceSynchronize());
        cudaHelpers::vecToSurface<<<32 * numSMs, 256>>>(imageBufferDenoised.surface, denoiseOutput, sceneRepresentation.sceneInfo.width, sceneRepresentation.sceneInfo.height);
        checkCudaErrors(cudaDeviceSynchronize());
    }



    actualSamples += spp;

    const auto availableSize = ImVec2{
            ImGui::GetWindowContentRegionMax().x - ImGui::GetWindowContentRegionMin().x,
            ImGui::GetWindowContentRegionMax().y - ImGui::GetWindowContentRegionMin().y,
    };

    auto drawTexture = [&availableSize](GLuint texture){
        glClear(GL_COLOR_BUFFER_BIT);
        glBindTexture(GL_TEXTURE_2D, texture);
        glBegin(GL_QUADS);
        glTexCoord2f(0, 0);
        glVertex2f(-1, -1);
        glTexCoord2f(1, 0);
        glVertex2f(1, -1);
        glTexCoord2f(1, 1);
        glVertex2f(1, 1);
        glTexCoord2f(0, 1);
        glVertex2f(-1, 1);
        glEnd();

        ImGui::Image(texture, ImVec2(float(availableSize[0]), float(availableSize[1])));

    };

    if(denoiserEnabled)
        drawTexture(imageBufferDenoised.texture);
    else
        drawTexture(imageBuffer.texture);

    return true;

}

__host__ void Scene::denoise() {

    clock_t startDenoise = clock();

    std::cout << "Starting denoise...";




    auto oidnDevice = oidn::newDevice();
    oidnDevice.commit();

    auto numDev = oidn::getNumPhysicalDevices();
    if(numDev == 0){
        std::cerr << "No OIDN devices found.\n";
    }

    const char* errorMessage;
    if (oidnDevice.getError(errorMessage) != oidn::Error::None){
        std::cerr << "OIDN Error: " << errorMessage << '\n';
    }

    int width = sceneRepresentation.sceneInfo.width;
    int height = sceneRepresentation.sceneInfo.height;



//    oidn::FilterRef filter = oidnDevice.newFilter("RT");
//    filter.setImage("color",  color.data(),  oidn::Format::Float3, width, height);
//    filter.setImage("albedo", albedos.data(), oidn::Format::Float3, width, height);
//    filter.setImage("normal", normals.data(), oidn::Format::Float3, width, height);
//    filter.setImage("output", hostImageBufferDenoised, oidn::Format::Float3, width, height);
//    filter.commit();
//
//    filter.execute();
//
//    if (oidnDevice.getError(errorMessage) != oidn::Error::None)
//        std::cerr << "Error: " << errorMessage << '\n';

    int devId = 0;
    int numSMs;
    checkCudaErrors(cudaDeviceGetAttribute(&numSMs, cudaDevAttrMultiProcessorCount, devId));
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    denoiser<<<blockSize, threadSize>>>(imageBuffer.featureBuffer, denoiseOutput, denoiseWeights, sceneRepresentation.sceneInfo.width, sceneRepresentation.sceneInfo.height);
    checkCudaErrors(cudaDeviceSynchronize());
    cudaHelpers::vecToSurface<<<32 * numSMs, 256>>>(imageBufferDenoised.surface, denoiseOutput, sceneRepresentation.sceneInfo.width, sceneRepresentation.sceneInfo.height);
    checkCudaErrors(cudaDeviceSynchronize());


    std::cout << "\rDenoising took " << ((double) (clock() - startDenoise)) / CLOCKS_PER_SEC << " seconds.\n";

}

struct FunctorIdentity {
    __device__ Vec3f operator()(const Vec3f &v) const {
        return v;
    }
};

__host__ void Scene::saveOutput() {

    std::cout << "Writing resulting image to disk...\n";

    if(!std::filesystem::exists("./data"))
        std::filesystem::create_directory("./data");

    const std::string pngPath = "./data/image.png";
    const std::string pngPathDenoised = "./data/imageDenoised.png";

    const std::string hdrPath = "./data/image.hdr";
    const std::string hdrPathDenoised = "./data/imageDenoised.hdr";

    Vec3f *hostImage;
    checkCudaErrors(cudaMallocManaged(&hostImage, sizeof(Vec3f) * sceneRepresentation.sceneInfo.width * sceneRepresentation.sceneInfo.height));
    unsigned int threadsPerBlock = 256;// Optimal number of threads per block
    unsigned int blocksPerGrid = (sceneRepresentation.sceneInfo.width * sceneRepresentation.sceneInfo.height + threadsPerBlock - 1) / threadsPerBlock;

    extractBufferInfo<BUFFERTYPE::MEAN><<<blocksPerGrid, threadsPerBlock>>>(imageBuffer.featureBuffer->color, hostImage, sceneRepresentation.sceneInfo.width, sceneRepresentation.sceneInfo.height, FunctorIdentity{});

//    Vec3f *hostImageDenoised;
//    checkCudaErrors(cudaMallocManaged(&hostImageDenoised, sizeof(Vec3f) * sceneRepresentation.sceneInfo.width * sceneRepresentation.sceneInfo.height));
//    extractBufferInfo<BUFFERTYPE::MEAN><<<blocksPerGrid, threadsPerBlock>>>(imageBufferDenoised.featureBuffer->color, hostImageDenoised, sceneRepresentation.sceneInfo.width, sceneRepresentation.sceneInfo.height, FunctorIdentity{});
//    checkCudaErrors(cudaDeviceSynchronize());

    Vec3f *hostImageDenoised = denoiseOutput;

    const bool didHDR = stbi_write_hdr(hdrPath.c_str(), sceneRepresentation.sceneInfo.width,
                                       sceneRepresentation.sceneInfo.height, 3, (float *) hostImage);
    assert(didHDR);

    const bool didHDRDenoised = stbi_write_hdr(hdrPathDenoised.c_str(), sceneRepresentation.sceneInfo.width,
                                       sceneRepresentation.sceneInfo.height, 3, (float *) hostImageDenoised);
    assert(didHDRDenoised);


    pngwriter png(sceneRepresentation.sceneInfo.width, sceneRepresentation.sceneInfo.height, 1., pngPath.c_str());
    pngwriter pngDenoised(sceneRepresentation.sceneInfo.width, sceneRepresentation.sceneInfo.height, 1.,
                          pngPathDenoised.c_str());


#pragma omp parallel for
    for(int j = 0; j < sceneRepresentation.sceneInfo.height; j++) {
        for(int i = 0; i < sceneRepresentation.sceneInfo.width; i++) {
            const int idx = j * sceneRepresentation.sceneInfo.width + i;
            png.plot(i + 1, sceneRepresentation.sceneInfo.height - j,
                     Warp::gammaCorrect(hostImage[idx][0]),
                     Warp::gammaCorrect(hostImage[idx][1]),
                     Warp::gammaCorrect(hostImage[idx][2]));
            pngDenoised.plot(i + 1, sceneRepresentation.sceneInfo.height - j,
                             Warp::gammaCorrect(hostImageDenoised[idx][0]),
                             Warp::gammaCorrect(hostImageDenoised[idx][1]),
                             Warp::gammaCorrect(hostImageDenoised[idx][2]));
        }
    }

    png.close();
    pngDenoised.close();

    std::cout << "Saving images completed.\n";
}

__host__ void Scene::step(float dt) noexcept {
    deviceCamera.translateRelative(cameraVelocity * dt);
//    deviceCamera.addVelocityRelative(cameraVelocity, dt);
    std::cout << std::flush;
}
__host__ OpenGLSharedBuffer::OpenGLSharedBuffer(size_t width, size_t height) : width(width),
                                                                             height(height) {

    checkCudaErrors(cudaMallocManaged(
            &featureBuffer,
            sizeof(FeatureBuffer)));

    new(featureBuffer) FeatureBuffer(width * height);


    glGenTextures(1, &texture);

    glBindTexture(GL_TEXTURE_2D, texture);

    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);


    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, width, height, 0, GL_RGB, GL_FLOAT, nullptr);


    checkCudaErrors(cudaGraphicsGLRegisterImage(&resource, texture, GL_TEXTURE_2D, cudaGraphicsRegisterFlagsNone));


    //TODO map multiple resources at once to batch all viewports
    checkCudaErrors(cudaGraphicsMapResources(1, &resource, nullptr));

    cudaArray_t cudaArray;
    checkCudaErrors(cudaGraphicsSubResourceGetMappedArray(&cudaArray, resource, 0, 0));

    cudaResourceDesc resDesc = {};
    resDesc.resType = cudaResourceTypeArray;
    resDesc.res.array.array = cudaArray;


    checkCudaErrors(cudaCreateSurfaceObject(&surface, &resDesc));


}
__host__ OpenGLSharedBuffer::~OpenGLSharedBuffer() {

    checkCudaErrors(cudaDestroySurfaceObject(surface));
    checkCudaErrors(cudaGraphicsUnmapResources(1, &resource, nullptr));

    checkCudaErrors(cudaGraphicsUnregisterResource(resource));
    glDeleteTextures(1, &texture);

    featureBuffer->~FeatureBuffer();
    checkCudaErrors(cudaFree(featureBuffer));
}
