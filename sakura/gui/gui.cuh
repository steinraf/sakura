//
// Created by steinraf on 04.02.25.
//

#pragma once

#include <Eigen/Dense>
#include <curand_kernel.h>

#include "imgui.h"

#include <GL/glew.h>
#include <GLFW/glfw3.h>

#include "../camera/camera.cuh"
#include "../common.cuh"


class GUI {
private:
public:
    explicit GUI(bool fullscreen);
    ~GUI();

    void loop(const Scene &scene, std::vector<Sensor> sensors);

    [[nodiscard]] Eigen::Vector2f getWindowSize() const;

private:
    bool fullscreen;
    GLFWwindow *window;

    const std::string applicationTitle = "Sakura";
    const std::string settingsTitle = "Settings";
};
