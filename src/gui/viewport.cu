//
// Created by steinraf on 07.04.25.
//

#include "viewport.cuh"
#include <cuda_gl_interop.h>

#include "../../src_old/cudaHelpers.cuh"
#include "../../src_old/scene/scene.h"
#include "../denoise/denoise.h"
#include "cuda_runtime.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

__host__ OpenGLSharedBuffer::OpenGLSharedBuffer(size_t width, size_t height) : width(width),
                                                                               height(height) {

    checkCudaErrors(cudaMallocManaged(
            &featureBuffer,
            sizeof(FeatureBuffer)));

    new(featureBuffer) FeatureBuffer(width * height);


    glGenTextures(1, &texture);

    //verify that texture is correctly created
    GLenum err = glGetError();
    if(err != GL_NO_ERROR) {
        std::cerr << "OpenGL error: " << err << std::endl;
        throw std::runtime_error("Failed to create OpenGL texture");
    }
    assert(texture != 0);



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


void Viewport::renderFrame(bool synchronize) {
    render();
    if(synchronize) checkCudaErrors(cudaDeviceSynchronize());
}
void OpenGLViewport::render() {
    auto dt = 1.f/std::min(ImGui::GetIO().Framerate, 1000.f);
    config.camera.translateRelative(cameraVelocity * dt);
    config.camera.rotateRelative(cameraRotation * dt);
    std::cout << std::flush;

    //TODO add correct tone mapping for live preview
    if(actualSamples < config.spp) {
        int devId = 0;
        int numSMs;
        checkCudaErrors(cudaDeviceGetAttribute(&numSMs, cudaDevAttrMultiProcessorCount, devId));
        checkCudaErrors(cudaGetLastError());
        checkCudaErrors(cudaDeviceSynchronize());

        assert(imageBuffer.featureBuffer);

        const unsigned int blockSizeX = 4, blockSizeY = 4;
        const dim3 threadSize{blockSizeX, blockSizeY};
        const dim3 blockSize(imageBuffer.width / blockSizeX + 1, imageBuffer.height / blockSizeY + 1);

        cudaHelpers::render<<<blockSize, threadSize>>>(config.camera, scene.getMeshAccelerationStructure(),
                                                       imageBuffer.width, imageBuffer.height, 1, config.maxRayDepth,
                                                       config.curandState, imageBuffer.featureBuffer);
        checkCudaErrors(cudaGetLastError());
        checkCudaErrors(cudaDeviceSynchronize());



        actualSamples += 1;

    } else {
        is_rendering_done = true;
    }
}

OpenGLViewport::OpenGLViewport(Scene &scene, OpenGLConfig&& config, std::string title)
    : imageBuffer(config.width, config.height), config(std::move(config)), scene(scene), title(std::move(title)) {

}
std::string OpenGLViewport::getTitle() const {
    return title;
}

bool OpenGLViewport::isDone() const {
    return is_rendering_done;
}

struct FunctorIdentity {
    __device__ Vec3f operator()(const Vec3f &v) const {
        return v;
    }
};

void OpenGLViewport::save() {
    if(!std::filesystem::exists("./data"))
        std::filesystem::create_directory("./data");

    const std::filesystem::path pngPath = "./data/image.png";
    const std::filesystem::path hdrPath = "./data/image.hdr";

    Vec3f *hostImage;
    checkCudaErrors(cudaMallocManaged(&hostImage, sizeof(Vec3f) * imageBuffer.width * imageBuffer.height));
    unsigned int threadsPerBlock = 256;// Optimal number of threads per block
    unsigned int blocksPerGrid = (imageBuffer.width * imageBuffer.height + threadsPerBlock - 1) / threadsPerBlock;

    extractBufferInfo<BUFFERTYPE::MEAN><<<blocksPerGrid, threadsPerBlock>>>(imageBuffer.featureBuffer->color, hostImage, imageBuffer.width, imageBuffer.height, FunctorIdentity{});

    [[maybe_unused]] const bool didHDR = stbi_write_hdr(hdrPath.c_str(), imageBuffer.width,
                                       imageBuffer.height, 3, (float *) hostImage);
    assert(didHDR);

    pngwriter png(imageBuffer.width, imageBuffer.height, 1., pngPath.c_str());

#pragma omp parallel for
    for(int j = 0; j < imageBuffer.height; j++) {
        for(int i = 0; i < imageBuffer.width; i++) {
            const int idx = j * imageBuffer.width + i;
            png.plot(i + 1, imageBuffer.height - j,
                     Warp::gammaCorrect(hostImage[idx][0]),
                     Warp::gammaCorrect(hostImage[idx][1]),
                     Warp::gammaCorrect(hostImage[idx][2]));
        }
    }

    png.close();

    std::cout << "Saving OpenGL Viewport " + title + " completed.\n";
}
void OpenGLViewport::clear() {
    imageBuffer.featureBuffer->clear();
}
void OpenGLViewport::drawTexture() {
    int numSMs;
    int devId = 0;

    checkCudaErrors(cudaDeviceGetAttribute(&numSMs, cudaDevAttrMultiProcessorCount, devId));
    checkCudaErrors(cudaDeviceSynchronize());
    cudaHelpers::bufferToSurface<<<32 * numSMs, 256>>>(imageBuffer.surface, imageBuffer.featureBuffer, imageBuffer.width, imageBuffer.height);
    checkCudaErrors(cudaDeviceSynchronize());

    const auto availableSize = ImVec2{
            ImGui::GetWindowContentRegionMax().x - ImGui::GetWindowContentRegionMin().x,
            ImGui::GetWindowContentRegionMax().y - ImGui::GetWindowContentRegionMin().y,
    };

    glClear(GL_COLOR_BUFFER_BIT);
    glBindTexture(GL_TEXTURE_2D, imageBuffer.texture);
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

    ImGui::Image(imageBuffer.texture, ImVec2(float(availableSize[0]), float(availableSize[1])));


}
void OpenGLViewport::handleInput() {
    static ImVec2 previousMousePos = ImGui::GetMousePos();


    Vector3f vCamera{0.f}, rotCamera{0.f};
    float linearSpeed = 1.f, angularSpeed = 0.2f;


    if(ImGui::IsKeyDown(ImGuiKey_W))
        vCamera[2] += 1.f;
    if(ImGui::IsKeyDown(ImGuiKey_S))
        vCamera[2] -= 1.f;

    if(ImGui::IsKeyDown(ImGuiKey_D))
        vCamera[0] += 1.f;
    if(ImGui::IsKeyDown(ImGuiKey_A))
        vCamera[0] -= 1.f;

    if(ImGui::IsKeyDown(ImGuiKey_UpArrow))
        rotCamera[0] += 1.f;
    if(ImGui::IsKeyDown(ImGuiKey_DownArrow))
        rotCamera[0] -= 1.f;
    if(ImGui::IsKeyDown(ImGuiKey_LeftArrow))
        rotCamera[1] += 1.f;
    if(ImGui::IsKeyDown(ImGuiKey_RightArrow))
        rotCamera[1] -= 1.f;

    if(ImGui::IsKeyDown(ImGuiKey_Space))
        vCamera[1] += 1.f;
    if(ImGui::IsKeyDown(ImGuiKey_LeftShift))
        vCamera[1] -= 1.f;

    ImVec2 mouseDelta = ImVec2{ImGui::GetMousePos().x - previousMousePos.x,
                               ImGui::GetMousePos().y - previousMousePos.y};

    if(ImGui::IsWindowHovered() and ImGui::IsMouseDown(ImGuiMouseButton_Left)){
        rotCamera += Vec3f{mouseDelta.y, mouseDelta.x, 0.f} * M_1_PI * 0.1f;
    }

    if(ImGui::IsKeyDown(ImGuiKey_LeftCtrl)){
        angularSpeed *= 0.1f;
        linearSpeed *= 0.1f;
    }

    if(ImGui::IsKeyDown(ImGuiKey_R))
        clear();


    cameraVelocity = vCamera * linearSpeed;
    cameraRotation = rotCamera * angularSpeed;

    if((vCamera.squaredNorm() != 0.f or rotCamera.squaredNorm() != 0.f) and not ImGui::IsKeyDown(ImGuiKey_R))
        clear();

    previousMousePos = ImGui::GetMousePos();
}
const FeatureBuffer *OpenGLViewport::getFeatureBuffer() const {
    return imageBuffer.featureBuffer;
}
Denoiser::Denoiser(std::shared_ptr<OpenGLViewport> _viewport, std::string title)
    : viewport(std::move(_viewport)), title(std::move(title)), denoiseWeights(nullptr), denoiseOutput(nullptr) {


}
std::string Denoiser::getTitle() const {
    return title;
}
bool Denoiser::isDone() const {
    return viewport->isDone();
}
void Denoiser::save() {

    //TODO potentially only re-render if denoiser has never been called
    auto start = std::chrono::high_resolution_clock::now();
    denoise();
    auto end = std::chrono::high_resolution_clock::now();
    std::cout << "Denoising took " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms\n";

    if(!std::filesystem::exists("./data"))
        std::filesystem::create_directory("./data");



    const std::string pngPathDenoised = "./data/imageDenoised.png";
    const std::string hdrPathDenoised = "./data/imageDenoised.hdr";


    Vec3f *hostImageDenoised = denoiseOutput;



    [[maybe_unused]] const bool didHDRDenoised = stbi_write_hdr(hdrPathDenoised.c_str(), viewport->imageBuffer.width,
                                               viewport->imageBuffer.height, 3, (float *) hostImageDenoised);
    assert(didHDRDenoised);


    pngwriter pngDenoised(viewport->imageBuffer.width, viewport->imageBuffer.height, 1.,
                          pngPathDenoised.c_str());


#pragma omp parallel for
    for(int j = 0; j < viewport->imageBuffer.height; j++) {
        for(int i = 0; i < viewport->imageBuffer.width; i++) {
            const int idx = j * viewport->imageBuffer.width + i;
            pngDenoised.plot(i + 1, viewport->imageBuffer.height - j,
                             Warp::gammaCorrect(hostImageDenoised[idx][0]),
                             Warp::gammaCorrect(hostImageDenoised[idx][1]),
                             Warp::gammaCorrect(hostImageDenoised[idx][2]));
        }
    }


    pngDenoised.close();

    std::cout << "Saving Denoised OpenGL Viewport " + title + " completed.\n";

}
void Denoiser::render() {
    viewport->render();
    denoise();
}
void Denoiser::clear() {

}
Denoiser::~Denoiser() {
    checkCudaErrors(cudaFree(denoiseWeights));
    checkCudaErrors(cudaFree(denoiseOutput));
}
void Denoiser::drawTexture() {
    int numSMs;
    int devId = 0;
    checkCudaErrors(cudaDeviceGetAttribute(&numSMs, cudaDevAttrMultiProcessorCount, devId));
    checkCudaErrors(cudaDeviceSynchronize());
    cudaHelpers::vecToSurface<<<32 * numSMs, 256>>>(viewport->imageBuffer.surface, denoiseOutput, viewport->imageBuffer.width, viewport->imageBuffer.height);
    checkCudaErrors(cudaDeviceSynchronize());

    const auto availableSize = ImVec2{
            ImGui::GetWindowContentRegionMax().x - ImGui::GetWindowContentRegionMin().x,
            ImGui::GetWindowContentRegionMax().y - ImGui::GetWindowContentRegionMin().y,
    };

    glClear(GL_COLOR_BUFFER_BIT);
    glBindTexture(GL_TEXTURE_2D, viewport->imageBuffer.texture);
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

    ImGui::Image(viewport->imageBuffer.texture, ImVec2(float(availableSize[0]), float(availableSize[1])));
}
void Denoiser::handleInput() {
    viewport->handleInput();
}
void Denoiser::denoise() {
    if(!denoiseOutput){
        checkCudaErrors(cudaMallocManaged(&denoiseWeights, sizeof(float) * viewport->scene.getDimensions().x * viewport->scene.getDimensions().y));
        checkCudaErrors(cudaMallocManaged(&denoiseOutput, sizeof(Vec3f) * viewport->scene.getDimensions().x * viewport->scene.getDimensions().y));
    }
    checkCudaErrors(cudaMemset(denoiseWeights, 0, sizeof(float) * viewport->imageBuffer.width * viewport->imageBuffer.height));
    checkCudaErrors(cudaMemset(denoiseOutput, 0, sizeof(Vec3f) * viewport->imageBuffer.width * viewport->imageBuffer.height));


    const unsigned int blockSizeX = 4, blockSizeY = 4;
    const dim3 threadSize{blockSizeX, blockSizeY};
    const dim3 blockSize(viewport->imageBuffer.width / blockSizeX + 1, viewport->imageBuffer.height / blockSizeY + 1);


    denoiser<<<blockSize, threadSize>>>(viewport->imageBuffer.featureBuffer, denoiseOutput, denoiseWeights, viewport->imageBuffer.width, viewport->imageBuffer.height);
    checkCudaErrors(cudaDeviceSynchronize());
    denoiseApplyWeights<<<blockSize, threadSize>>>(viewport->imageBuffer.featureBuffer, denoiseOutput, denoiseWeights, viewport->imageBuffer.width, viewport->imageBuffer.height);
    checkCudaErrors(cudaDeviceSynchronize());

}
OpenGLViewport::OpenGLConfig::OpenGLConfig(Camera camera, int spp, int maxRayDepth, int width, int height)
    : camera(std::move(camera)), spp(spp), maxRayDepth(maxRayDepth), width(width), height(height), curandState(nullptr) {

    checkCudaErrors(cudaMalloc(&curandState, sizeof(decltype(*curandState)) * width * height));

    const unsigned int blockSizeX = 4, blockSizeY = 4;
    const dim3 threadSize{blockSizeX, blockSizeY};
    const dim3 blockSize(width / blockSizeX + 1, height / blockSizeY + 1);


    cudaHelpers::initRng<<<blockSize, threadSize>>>(width, height, curandState);
    checkCudaErrors(cudaGetLastError());

    checkCudaErrors(cudaDeviceSynchronize());

}
