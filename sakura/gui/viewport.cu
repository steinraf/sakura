//
// Created by steinraf on 24.02.25.
//

#include "../integrator/integrators.cuh"
#include "viewport.cuh"

#include <cassert>

#include <curand_kernel.h>
#include <iomanip>

#include "../denoise/denoise.cuh"
#include "../scene/scene.cuh"

static constexpr int RNG_SEED = 42;


__global__ void initializeRNG(curandState *rngStates, size_t numSamples) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    if(idx >= numSamples) return;

    curand_init(RNG_SEED, idx, 0, &rngStates[idx]);
}


__global__ void clearFeatureBuffer(FeatureBuffer *buffer) {

    auto idx = blockIdx.x * blockDim.x + threadIdx.x;

    if(idx >= buffer->numElements) {
        printf("Terminating early for idx %d\n", idx);
        return;
    }

    buffer->color[idx] = {};
    buffer->normal[idx] = {};
    buffer->position[idx] = {};
    buffer->albedo[idx] = {};
    buffer->uv[idx] = {};
}

__host__ FeatureBuffer::FeatureBuffer(size_t numElements) : numElements(numElements), color(nullptr), normal(nullptr), position(nullptr), albedo(nullptr), uv(nullptr) {
    checkCudaErrors(cudaMallocManaged(&color, numElements * sizeof(Statistic<Eigen::Vector3f>)));
    checkCudaErrors(cudaMallocManaged(&normal, numElements * sizeof(Statistic<Eigen::Vector3f>)));
    checkCudaErrors(cudaMallocManaged(&position, numElements * sizeof(Statistic<Eigen::Vector3f>)));
    checkCudaErrors(cudaMallocManaged(&albedo, numElements * sizeof(Statistic<Eigen::Vector3f>)));
    checkCudaErrors(cudaMallocManaged(&uv, numElements * sizeof(Statistic<Eigen::Vector3f>)));
    clear();
}

__host__ FeatureBuffer::~FeatureBuffer() {
    checkCudaErrors(cudaFree(color));
    checkCudaErrors(cudaFree(normal));
    checkCudaErrors(cudaFree(position));
    checkCudaErrors(cudaFree(albedo));
    checkCudaErrors(cudaFree(uv));
}
void FeatureBuffer::clear() {
    int threadsPerBlock = 256;
    size_t blocksPerGrid = (numElements + threadsPerBlock - 1) / threadsPerBlock;

    clearFeatureBuffer<<<blocksPerGrid, threadsPerBlock>>>(this);
    checkCudaErrors(cudaDeviceSynchronize());
}


OpenGLViewport::OpenGLViewport(const Scene &scene, unsigned int width, unsigned int height, std::string title, Camera camera, int spp)
    : scene(scene), size({width, height}), title(std::move(title)),
      texture(0), resource(nullptr), featureBuffer(nullptr), surface(), camera(std::move(camera)), rngStates(nullptr), samplesPerPixel(spp) {

    checkCudaErrors(cudaMallocManaged(
            &rngStates,
            width * height * sizeof(curandState)));

    checkCudaErrors(cudaMallocManaged(
            &featureBuffer,
            sizeof(FeatureBuffer)));

    new(featureBuffer) FeatureBuffer(width * height);

    //    *featureBuffer = FeatureBuffer(width * height);

    unsigned int threadsPerBlock = 256;// Optimal number of threads per block
    unsigned int blocksPerGrid = (width * height + threadsPerBlock - 1) / threadsPerBlock;

    initializeRNG<<<blocksPerGrid, threadsPerBlock>>>(
            rngStates, width * height);

    checkCudaErrors(cudaDeviceSynchronize());

    glGenTextures(1, &texture);

    glBindTexture(GL_TEXTURE_2D, texture);

    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);


    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, safe_uint_to_int(width), safe_uint_to_int(height), 0, GL_RGB, GL_FLOAT, nullptr);


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

void OpenGLViewport::render() {


    t += dt;
    //    float circleScale = 1.0;
    //    translateCamera(dt * Eigen::Vector3f{circleScale * std::sin(t), 0.0, circleScale * std::cos(t)});

    handleUserInput();

    ImGui::Begin(title.c_str(), nullptr, ImGuiWindowFlags_NoDecoration);// , nullptr, ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_NoSavedSettings
                                                                        //    ImGui::SetWindowSize(size);


    scene.render(surface, featureBuffer, camera, rngStates, size, samplesPerPixel);


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

    ImGui::Image(texture, ImVec2(float(size[0]), float(size[1])));

    ImGui::End();
}
void OpenGLViewport::translateCamera(const Eigen::Vector3f &translation) {
    camera.translate(translation);
    featureBuffer->clear();
}

void OpenGLViewport::translateCameraRelative(const Eigen::Vector3f &translation) {
    camera.relativeTranslate(translation);
    featureBuffer->clear();
}

void OpenGLViewport::generateSettings() {
    ImGui::SliderInt("Samples per Pixel", &samplesPerPixel, 1, 16);
}
OpenGLViewport::~OpenGLViewport() {

    checkCudaErrors(cudaDestroySurfaceObject(surface));
    checkCudaErrors(cudaGraphicsUnmapResources(1, &resource, nullptr));

    checkCudaErrors(cudaFree(rngStates));
    checkCudaErrors(cudaFree(featureBuffer));

    checkCudaErrors(cudaGraphicsUnregisterResource(resource));
    glDeleteTextures(1, &texture);
}
std::string OpenGLViewport::getTitle() const {
    return title;
}
void OpenGLViewport::generateDebugInformation() {
}
Eigen::Vector2f OpenGLViewport::getWindowSize() const {
    return Eigen::Vector2f{size[0], size[1]};
}
FeatureBuffer *OpenGLViewport::getFeatureBuffer() const {
    return featureBuffer;
}
void OpenGLViewport::handleUserInput() {
    constexpr float cameraVel = 10.0;
    if(ImGui::IsKeyPressed(ImGuiKey_W)) {
        translateCameraRelative(cameraVel * Eigen::Vector3f{0, 0, dt});
    }
    if(ImGui::IsKeyPressed(ImGuiKey_S)) {
        translateCameraRelative(cameraVel * Eigen::Vector3f{0, 0, -dt});
    }
    if(ImGui::IsKeyPressed(ImGuiKey_Space)) {
        translateCameraRelative(cameraVel * Eigen::Vector3f{0, dt, 0});
    }
    if(ImGui::IsKeyPressed(ImGuiKey_LeftShift)) {
        translateCameraRelative(cameraVel * Eigen::Vector3f{0, -dt, 0});
    }
    if(ImGui::IsKeyPressed(ImGuiKey_D)) {
        translateCameraRelative(cameraVel * Eigen::Vector3f{dt, 0, 0});
    }
    if(ImGui::IsKeyPressed(ImGuiKey_A)) {
        translateCameraRelative(cameraVel * Eigen::Vector3f{-dt, 0, 0});
    }
}


Denoiser::Denoiser(FeatureBuffer *buffer, unsigned int width, unsigned int height, std::string title)
    : buffer(buffer), size(width, height), title(std::move(title)), texture(0), resource(nullptr), surface() {

    glGenTextures(1, &texture);

    glBindTexture(GL_TEXTURE_2D, texture);

    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, safe_uint_to_int(width), safe_uint_to_int(height), 0, GL_RGB, GL_FLOAT, nullptr);


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
Denoiser::~Denoiser() {
    checkCudaErrors(cudaDestroySurfaceObject(surface));
    checkCudaErrors(cudaGraphicsUnmapResources(1, &resource, nullptr));

    checkCudaErrors(cudaGraphicsUnregisterResource(resource));
    glDeleteTextures(1, &texture);
}
std::string Denoiser::getTitle() const {
    return title;
}
void Denoiser::generateDebugInformation() {
}
void Denoiser::generateSettings() {
}
void Denoiser::render() {

    ImGui::Begin(title.c_str(), nullptr, ImGuiWindowFlags_NoDecoration);// , nullptr, ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_NoSavedSettings
                                                                        //    ImGui::SetWindowSize(size);

    int devId = 0;
    int numSMs;
    checkCudaErrors(cudaDeviceGetAttribute(&numSMs, cudaDevAttrMultiProcessorCount, devId));

    denoise<<<32 * numSMs, 256>>>(surface, buffer, size[0], size[1]);
    checkCudaErrors(cudaDeviceSynchronize());

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

    ImGui::Image(texture, ImVec2(float(size[0]), float(size[1])));

    ImGui::End();
}
void Renderable::renderFrame(bool synchronize) {
    auto renderStart = std::chrono::high_resolution_clock::now();
    render();
    if(synchronize) checkCudaErrors(cudaDeviceSynchronize());

    auto renderEnd = std::chrono::high_resolution_clock::now();
    double deltaTime = static_cast<double>(std::chrono::duration_cast<Duration>(renderEnd - renderStart).count());

    frameTimes.addElement(deltaTime);
}
void Renderable::generateTimingInformation() {


    using conversionRatio = std::ratio_divide<Duration::period, std::chrono::milliseconds::period>;

    auto convert = [](double duration) {
        return duration * conversionRatio::num / conversionRatio::den;
    };

    auto convertSq = [](double duration) {
        using conversionRatioSq = std::ratio_multiply<conversionRatio, conversionRatio>;
        return duration * conversionRatioSq::num / conversionRatioSq::den;
    };


    auto mean = convert(frameTimes.getMean());
    auto var = convertSq(frameTimes.getVariance());            // convert twice because variance is squared
    auto samplevar = convertSq(frameTimes.getSampleVariance());// convert twice because sample variance is squared
    auto elemCount = frameTimes.getNumElements();

    std::stringstream ss;


#define FMT std::setprecision(3) << std::setw(8) << std::setfill(' ') << std::fixed

    ss << FMT << mean << " ms avg | "
       << FMT << var << " ms variance avg | "
       << FMT << samplevar << " ms sample variance avg | "
       << FMT << elemCount << "#";

    std::string str = ss.str();

    ImGui::Text("%s", str.c_str());

    //    ImGui::Text("%f ms avg | %f ms variance", mean, var);
}
