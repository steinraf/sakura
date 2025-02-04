//
// Created by steinraf on 04.02.25.
//

#pragma once

#include <Eigen/Dense>

#include "imgui.h"

#include <GL/glew.h>
#include <GLFW/glfw3.h>

#include "../scene/scene.cuh"

class OpenGLViewport{
public:
    OpenGLViewport(int width, int height, std::string title, Camera camera);

    void renderFrame(const Scene& scene);

private:
    ImVec2 size;
    std::string title;
    GLuint texture;
    cudaGraphicsResource_t resource;

    Camera camera;
};



class GUI {
private:
public:
    explicit GUI(bool fullscreen);
    ~GUI();

    void loop();

    [[nodiscard]] Eigen::Vector2f getWindowSize() const;

private:
    bool fullscreen;
    GLFWwindow *window;

    const std::string applicationTitle = "Sakura";

};

