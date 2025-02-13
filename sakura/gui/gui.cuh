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

class OpenGLViewport {
public:
    OpenGLViewport(int width, int height, std::string title, Camera camera, int spp = 1);

    ~OpenGLViewport();

    void renderFrame(const Scene &scene);
    void translateCamera(const Eigen::Vector3f &translation);

    void createSamplesPerPixelSlider(int min = 1, int max = 16);

    [[nodiscard]] std::string getTitle() const;

private:
    ImVec2 size;
    std::string title;
    GLuint texture;
    cudaGraphicsResource_t resource;
    cudaSurfaceObject_t surface;

    Camera camera;
    curandState *rngStates;
    int samplesPerPixel;
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
