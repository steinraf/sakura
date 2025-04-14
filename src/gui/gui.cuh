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
    GUIConfig() = delete;
    explicit GUIConfig(std::string applicationTitle, bool fullscreen) : applicationTitle(std::move(applicationTitle)), fullscreen(fullscreen) {}
};

class GUI {
public:
    explicit GUI(const GUIConfig &config);
    ~GUI();

    CPU_ONLY void loop(Scene& scene);
    CPU_ONLY void setViewports(const std::vector<std::shared_ptr<Viewport>> &viewports);


private:
    GLFWwindow *window;
    GUIConfig config;

    std::unordered_map<std::string, std::shared_ptr<Viewport>> viewports;

};
