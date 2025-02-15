//
// Created by steinraf on 04.02.25.
//

#pragma once

#include <Eigen/Dense>
#include <curand_kernel.h>

#include "imgui.h"

#include <GL/glew.h>
#include <GLFW/glfw3.h>

#include "../scene/scene.cuh"


template<typename T>
class Statistic {
public:
    __host__ __device__ Statistic() : numElements(0), mean({}), variance({}) {}

    // Welford's online algorithm to incrementally calculate variance
    __host__ __device__ void addElement(T element);

    [[nodiscard]] __host__ __device__ T getMean() const {
        return mean;
    }

    [[nodiscard]] __host__ __device__ T getVariance() const {
        return variance / numElements;
    }

    [[nodiscard]] __host__ __device__ T getSampleVariance() const {
        return variance / (numElements - 1);
    }

    [[nodiscard]] __host__ __device__ size_t getNumElements() const {
        return numElements;
    }

    __host__ __device__ void clear() {
        numElements = 0;
        mean = {};
        variance = {};
    }

private:
    size_t numElements;
    T mean;
    T variance;
};

template<typename T>
__host__ __device__ void Statistic<T>::addElement(T element) {
    numElements++;
    T delta = element - mean;
    mean += delta / numElements;
    if constexpr(std::is_same_v<T, Eigen::Vector3f>) {
        variance += delta.cwiseProduct(element - mean);
    } else {
        variance += delta * (element - mean);
    }
}

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
};

class Renderable {
public:
    virtual ~Renderable() = default;

    virtual void renderFrame() = 0;
    virtual void generateSettings() = 0;
    virtual void generateDebugInformation() = 0;
    [[nodiscard]] virtual std::string getTitle() const = 0;

private:
};

class OpenGLViewport : public Renderable {
public:
    OpenGLViewport(const Scene &scene, int width, int height, std::string title, Camera camera, int spp = 1);

    ~OpenGLViewport() override;

    void renderFrame() override;
    void translateCamera(const Eigen::Vector3f &translation);

    void generateSettings() override;
    void generateDebugInformation() override;

    [[nodiscard]] std::string getTitle() const override;
    [[nodiscard]] Eigen::Vector2f getWindowSize() const;
    [[nodiscard]] FeatureBuffer *getFeatureBuffer() const;

private:
    // Every camera is bound to a scene
    const Scene &scene;
    ImVec2 size;
    std::string title;
    GLuint texture;
    cudaGraphicsResource_t resource;
    FeatureBuffer *featureBuffer;
    cudaSurfaceObject_t surface;

    Camera camera;
    curandState *rngStates;
    int samplesPerPixel;
};


template<typename F>
class BufferVisualizer : public Renderable {
public:
    BufferVisualizer(Statistic<Eigen::Vector3f> *stat, std::string title, int width, int height, F f);

    ~BufferVisualizer() override;

    void renderFrame() override;
    void generateSettings() override;
    void generateDebugInformation() override;

    [[nodiscard]] std::string getTitle() const override;

private:
    const Statistic<Eigen::Vector3f> *stat;
    ImVec2 size;
    std::string title;
    GLuint texture{};
    cudaGraphicsResource_t resource{};
    cudaSurfaceObject_t surface{};
    F f;
};


class Denoiser : public Renderable {
public:
    Denoiser(FeatureBuffer *buffer, int width, int height, std::string title);

    ~Denoiser() override;

    void renderFrame() override;
    void generateSettings() override;
    void generateDebugInformation() override;

    [[nodiscard]] std::string getTitle() const override;

private:
    FeatureBuffer *buffer;
    ImVec2 size;
    std::string title;
    GLuint texture;
    cudaGraphicsResource_t resource;
    cudaSurfaceObject_t surface;
};

class GUI {
private:
public:
    explicit GUI(bool fullscreen);
    ~GUI();

    void loop(const Scene &scene);

    [[nodiscard]] Eigen::Vector2f getWindowSize() const;

private:
    bool fullscreen;
    GLFWwindow *window;

    const std::string applicationTitle = "Sakura";
    const std::string settingsTitle = "Settings";
};
