//
// Created by steinraf on 05.04.25.
//

#pragma once

#include "../../src_old/scene/scene.h"
#include <GLFW/glfw3.h>
#include <string>


struct GUIConfig {
    std::string applicationTitle = "Sakura Path Tracer";
    bool fullscreen = false;
};

class GUI {
public:
    explicit GUI(const GUIConfig &config);
    ~GUI();

    CPU_ONLY void loop(Scene& scene);




private:
    GLFWwindow *window;
    GUIConfig config;
};
