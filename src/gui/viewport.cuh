//
// Created by steinraf on 07.04.25.
//

#pragma once

#include "../camera/camera.cuh"
#include "../common.h"
#include "../denoise/featureBuffer.cuh"
#include <GL/glew.h>
#include <curand_kernel.h>
#include <memory>
#include <string>



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

    void clear() override;
    void save() override;

    void drawTexture() override;

    void handleInput() override;



private:
    friend class Denoiser;

    void render() override;

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