//
// Created by steinraf on 07.04.25.
//

#pragma once

#include "../camera/camera.cuh"
#include "../common.h"
#include "../denoise/featureBuffer.cuh"
#include "imgui.h"
#include <GL/glew.h>
#include <curand_kernel.h>
#include <memory>
#include <string>
#include <utility>
#include <thrust/functional.h>



class Viewport{
public:
    //Default virtual destructor
    virtual ~Viewport() = default;

    void renderFrame(bool synchronize=true);

    [[nodiscard]] virtual std::string getTitle() const = 0;
    [[nodiscard]] virtual bool isDone() const = 0;

    virtual void handleInput() = 0;

    virtual void clear() = 0;
    virtual void save() = 0;

    virtual void drawTexture() = 0;

private:
    virtual void render() = 0;
};


struct OpenGLSharedBuffer{
    GLuint texture{};
    cudaGraphicsResource_t resource{};
    FeatureBuffer *featureBuffer{};
    cudaSurfaceObject_t surface{};
    size_t width, height;

    CPU_ONLY explicit OpenGLSharedBuffer(size_t width, size_t height);
    CPU_ONLY ~OpenGLSharedBuffer();
    OpenGLSharedBuffer(const OpenGLSharedBuffer&) = delete;
    OpenGLSharedBuffer(OpenGLSharedBuffer&&) = default;
    OpenGLSharedBuffer& operator=(const OpenGLSharedBuffer&) = delete;
    OpenGLSharedBuffer& operator=(OpenGLSharedBuffer&&) = default;
};


class OpenGLViewport : public Viewport {
public:
    struct OpenGLConfig{
        Camera camera;
        int spp;
        int maxRayDepth;
        int width, height;
        curandState *curandState;
        OpenGLConfig() = delete;
        OpenGLConfig(Camera camera, int spp, int maxRayDepth, int width, int height);
    };

    OpenGLViewport(Scene &scene, OpenGLConfig&& config, std::string title);
    ~OpenGLViewport() override = default;
    OpenGLViewport(const OpenGLViewport&) = delete;
    OpenGLViewport(OpenGLViewport&&) = default;
    OpenGLViewport& operator=(const OpenGLViewport&) = delete;
    OpenGLViewport& operator=(OpenGLViewport&&) = delete;


    [[nodiscard]] std::string getTitle() const override;
    [[nodiscard]] bool isDone() const override;
    [[nodiscard]] const FeatureBuffer *getFeatureBuffer() const;


    void clear() override;
    void save() override;

    void drawTexture() override;

    void handleInput() override;



private:
    friend class Denoiser;

    template<BUFFERTYPE B, typename F>
    friend class BufferVisualizer;

    void render() override;


    OpenGLSharedBuffer imageBuffer;


    OpenGLConfig config;

    int actualSamples = 0;

    Scene &scene;
    std::string title;

    bool is_rendering_done = false;

    Vec3f cameraVelocity{0.f}, cameraRotation{0.f};



};

class Denoiser : public Viewport {
public:
    Denoiser(std::shared_ptr<OpenGLViewport> viewport, std::string title);
    ~Denoiser() override;

    [[nodiscard]] std::string getTitle() const override;
    [[nodiscard]] bool isDone() const override;

    void clear() override;
    void save() override;

    void drawTexture() override;

    void handleInput() override;

private:
    void render() override;

    void denoise();


    std::shared_ptr<OpenGLViewport> viewport;
    std::string title;

    float *denoiseWeights;
    Vec3f *denoiseOutput;

};

template<BUFFERTYPE B, typename F>
class BufferVisualizer : public Viewport {
public:
    BufferVisualizer(std::shared_ptr<OpenGLViewport> viewport, const Statistic<Vec3f> *stat, std::string title, auto &&f);
    ~BufferVisualizer() override;

    [[nodiscard]] std::string getTitle() const override {return title;};
    [[nodiscard]] bool isDone() const override {return viewport->isDone();};

    void clear() override { viewport->clear(); };
    void save() override;

    void drawTexture() override;

    void handleInput() override;
private:
    void render() override;

    std::shared_ptr<OpenGLViewport> viewport;
    const Statistic<Vec3f> *stat;
    std::string title;

    F f;

};
template<BUFFERTYPE B, typename F>
void BufferVisualizer<B, F>::render() {
    viewport->render();


}
template<BUFFERTYPE B, typename F>
void BufferVisualizer<B, F>::handleInput() {
    viewport->handleInput();
}


template<BUFFERTYPE B, typename F>
BufferVisualizer<B, F>::BufferVisualizer(std::shared_ptr<OpenGLViewport> viewport, const Statistic<Vec3f> *stat, std::string title, auto &&f)
    : viewport(std::move(viewport)), stat(stat), title(std::move(title)), f(std::forward<F>(f)) {
}

template<BUFFERTYPE B, typename F>
BufferVisualizer<B, F>::~BufferVisualizer() {

}


template<BUFFERTYPE B, typename F>
void BufferVisualizer<B, F>::save() {
    //TODO save buffer to image
    assert(false);
}

template<BUFFERTYPE B>
__global__ void renderBuffer(const Statistic<Vec3f> *stat, cudaSurfaceObject_t surface, int width, int height, auto f) {

    size_t pixelIndex = blockIdx.x * blockDim.x + threadIdx.x;

    if(pixelIndex >= width * height) return;

    size_t x = pixelIndex % width, y = pixelIndex / width;


    auto color = [&]() -> Vec3f {
        if constexpr(B == BUFFERTYPE::MEAN) return f(stat[pixelIndex].getMean());
        if constexpr(B == BUFFERTYPE::VARIANCE) return f(stat[pixelIndex].getVariance());
        if constexpr(B == BUFFERTYPE::SAMPLEVARIANCE) return f(stat[pixelIndex].getSampleVariance());
        if constexpr(B == BUFFERTYPE::NUM_ELEMENTS) return f(Vec3f{float(stat[pixelIndex].getNumElements()), float(stat[pixelIndex].getNumElements()), float(stat[pixelIndex].getNumElements())});
        else
            return Vec3f{-1, -1, -1};
    }();

    uchar4 color4 = make_uchar4(color[0] * 255,
                                color[1] * 255,
                                color[2] * 255, 255);


    surf2Dwrite(color4, surface, x * sizeof(uchar4), y);
}


template<BUFFERTYPE B, typename F>
void BufferVisualizer<B, F>::drawTexture() {

    unsigned int threadsPerBlock = 256;// Optimal number of threads per block
    unsigned int blocksPerGrid = (viewport->config.width * viewport->config.height + threadsPerBlock - 1) / threadsPerBlock;

    renderBuffer<B><<<blocksPerGrid, threadsPerBlock>>>(stat, viewport->imageBuffer.surface, viewport->config.width, viewport->config.height, f);
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