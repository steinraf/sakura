//
// Created by steinraf on 24.02.25.
//

#pragma once

#include <chrono>
#include <string>

#include <Eigen/Dense>

#include <GL/glew.h>
#include <curand_kernel.h>
#include <imgui.h>

#include "../camera/camera.cuh"
#include "../common.cuh"

#include <cuda_gl_interop.h>


enum class BUFFERTYPE {
    MEAN,
    VARIANCE,
    SAMPLEVARIANCE,
    NUM_ELEMENTS
};


template<BUFFERTYPE B>
__global__ void renderBuffer(const Statistic<Eigen::Vector3f> *stat, cudaSurfaceObject_t surface, int width, int height, auto /* Vector Transformation */ f);

__global__ void clearFeatureBuffer(FeatureBuffer *buffer);

struct FeatureBuffer {
    FeatureBuffer() = delete;
    explicit __host__ FeatureBuffer(size_t numElements);
    __host__ ~FeatureBuffer();

    void __host__ clear();

    size_t numElements;
    Statistic<Eigen::Vector3f> *color;
    Statistic<Eigen::Vector3f> *normal;
    Statistic<Eigen::Vector3f> *position;
    Statistic<Eigen::Vector3f> *albedo;
    Statistic<Eigen::Vector3f> *uv;
};

class Renderable {
public:
    using Duration = std::chrono::nanoseconds;

    virtual ~Renderable() = default;

    void renderFrame(bool synchronize = true);
    void generateTimingInformation();

    virtual void generateSettings() = 0;
    virtual void generateDebugInformation() = 0;
    [[nodiscard]] virtual std::string getTitle() const = 0;

private:
    Statistic<double> frameTimes;
    virtual void render() = 0;
};

class OpenGLViewport : public Renderable {
public:
    OpenGLViewport(const Scene &scene, unsigned int width, unsigned int height, std::string title, Camera camera, int spp = 1);

    ~OpenGLViewport() override;


    void translateCamera(const Eigen::Vector3f &translation);
    // (Right | Up | Forward )
    void translateCameraRelative(const Eigen::Vector3f &translation);
    void handleUserInput();

    void generateSettings() override;
    void generateDebugInformation() override;

    [[nodiscard]] std::string getTitle() const override;
    [[nodiscard]] Eigen::Vector2f getWindowSize() const;
    [[nodiscard]] FeatureBuffer *getFeatureBuffer() const;

private:
    void render() override;


    // Every camera is bound to a scene
    const Scene &scene;
    Eigen::Vector2<unsigned int> size;
    std::string title;
    GLuint texture;
    cudaGraphicsResource_t resource;
    FeatureBuffer *featureBuffer;
    cudaSurfaceObject_t surface;

    Camera camera;
    curandState *rngStates;
    int samplesPerPixel;

    float t = 0.0;
    const float dt = 0.01;
};


template<typename F, BUFFERTYPE B>
class BufferVisualizer : public Renderable {
public:
    BufferVisualizer(Statistic<Eigen::Vector3f> *stat, std::string title, unsigned int width, unsigned int height, F f);

    ~BufferVisualizer() override;


    void generateSettings() override;
    void generateDebugInformation() override;

    [[nodiscard]] std::string getTitle() const override;

private:
    void render() override;

    const Statistic<Eigen::Vector3f> *stat;
    Eigen::Vector2<unsigned int> size;
    std::string title;
    GLuint texture{};
    cudaGraphicsResource_t resource{};
    cudaSurfaceObject_t surface{};
    F f;
};


class Denoiser : public Renderable {
public:
    Denoiser(FeatureBuffer *buffer, unsigned int width, unsigned int height, std::string title);

    ~Denoiser() override;

    void generateSettings() override;
    void generateDebugInformation() override;

    [[nodiscard]] std::string getTitle() const override;

private:
    void render() override;

    FeatureBuffer *buffer;
    Eigen::Vector2<unsigned int> size;
    std::string title;
    GLuint texture;
    cudaGraphicsResource_t resource;
    cudaSurfaceObject_t surface;
};


template<typename F, BUFFERTYPE B>
BufferVisualizer<F, B>::BufferVisualizer(Statistic<Eigen::Vector3f> *stat, std::string title, unsigned int width, unsigned int height, F f)
    : stat(stat), size(width, height), title(std::move(title)), f(f) {


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

template<typename F, BUFFERTYPE B>
BufferVisualizer<F, B>::~BufferVisualizer() {
    checkCudaErrors(cudaDestroySurfaceObject(surface));
    checkCudaErrors(cudaGraphicsUnmapResources(1, &resource, nullptr));

    checkCudaErrors(cudaGraphicsUnregisterResource(resource));
    glDeleteTextures(1, &texture);
}

template<typename F, BUFFERTYPE B>
std::string BufferVisualizer<F, B>::getTitle() const {
    return title;
}

template<typename F, BUFFERTYPE B>
void BufferVisualizer<F, B>::generateDebugInformation() {
}
template<typename F, BUFFERTYPE B>
void BufferVisualizer<F, B>::generateSettings() {
}
template<typename F, BUFFERTYPE B>
void BufferVisualizer<F, B>::render() {

    ImGui::Begin(title.c_str(), nullptr, ImGuiWindowFlags_NoDecoration);// , nullptr, ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_NoSavedSettings
                                                                        //    ImGui::SetWindowSize(size);

    //TODO abstract kernel launches
    unsigned int threadsPerBlock = 256;// Optimal number of threads per block
    unsigned int blocksPerGrid = (size[0] * size[1] + threadsPerBlock - 1) / threadsPerBlock;

    renderBuffer<B><<<blocksPerGrid, threadsPerBlock>>>(stat, surface, size[0], size[1], f);
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

    ImGui::Image(texture, ImVec2(size[0], size[1]));

    ImGui::End();
}


template<BUFFERTYPE B>
__global__ void renderBuffer(const Statistic<Eigen::Vector3f> *stat, cudaSurfaceObject_t surface, int width, int height, auto f) {

    size_t pixelIndex = blockIdx.x * blockDim.x + threadIdx.x;

    if(pixelIndex >= width * height) return;

    size_t x = width - 1 - pixelIndex % width, y = pixelIndex / width;


    auto color = [&]() -> Eigen::Vector3f {
        if constexpr(B == BUFFERTYPE::MEAN) return f(stat[pixelIndex].getMean());
        if constexpr(B == BUFFERTYPE::VARIANCE) return f(stat[pixelIndex].getVariance());
        if constexpr(B == BUFFERTYPE::SAMPLEVARIANCE) return f(stat[pixelIndex].getSampleVariance());
        if constexpr(B == BUFFERTYPE::NUM_ELEMENTS) return f(Eigen::Vector3f{float(stat[pixelIndex].getNumElements()), float(stat[pixelIndex].getNumElements()), float(stat[pixelIndex].getNumElements())});
        else
            return Eigen::Vector3f{-1, -1, -1};
    }();

    uchar4 color4 = make_uchar4(color[0] * 255,
                                color[1] * 255,
                                color[2] * 255, 255);


    surf2Dwrite(color4, surface, x * sizeof(uchar4), y);
}
