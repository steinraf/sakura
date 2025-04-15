//
// Created by steinraf on 05.04.25.
//

#include "gui.cuh"

#include "imgui.h"

#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"
#include "imgui_internal.h"
#include "viewport.cuh"

#include <GL/glew.h>
#include <thread>

#include <iostream>

static void glfw_error_callback(int error, const char* description){
    fprintf(stderr, "Glfw Error %d: %s\n", error, description);
}

GUI::GUI(const GUIConfig &config) : config(config), viewports() {
    glfwSetErrorCallback(glfw_error_callback);

    if(!glfwInit()) {
        std::cerr << "Error while initializing glfw. Exiting.\n";
        exit(1);
    }

    const char *glsl_version = "#version 130";
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 0);

    window = [&]() -> GLFWwindow * {
        if(config.fullscreen) {
            const auto monitor = glfwGetPrimaryMonitor();
            const auto mode = glfwGetVideoMode(monitor);
            return glfwCreateWindow(mode->width, mode->height, config.applicationTitle.c_str(), monitor, nullptr);
        } else {
            int width, height;
            glfwGetMonitorWorkarea(glfwGetPrimaryMonitor(), nullptr, nullptr, &width, &height);
            return glfwCreateWindow(width, height, config.applicationTitle.c_str(), nullptr, nullptr);
        }
    }();

    if(window == nullptr) {
        std::cerr << "Window creation failed. Exiting.\n";
        exit(1);
    }
    glfwMakeContextCurrent(window);
    glfwSwapInterval(1);

    //ImGui Setup
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO &io = ImGui::GetIO();
    (void) io;
    io.ConfigFlags |= ImGuiConfigFlags_DockingEnable;
    io.ConfigFlags |= ImGuiConfigFlags_ViewportsEnable;
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableGamepad;
    io.IniFilename = nullptr;

    ImGui::StyleColorsDark();

    ImGuiStyle &style = ImGui::GetStyle();
    if(io.ConfigFlags & ImGuiConfigFlags_ViewportsEnable) {
        style.WindowRounding = 0.0f;
        style.Colors[ImGuiCol_WindowBg].w = 1.0f;
    }

    ImGui_ImplGlfw_InitForOpenGL(window, true);
    ImGui_ImplOpenGL3_Init(glsl_version);

    if(glewInit() != GLEW_OK) {
        std::cerr << "Glew not correctly initialized. Exiting.\n";
        exit(1);
    }
}


CPU_ONLY void GUI::loop(Scene &scene) {
    auto start = std::chrono::high_resolution_clock::now();
    while (!glfwWindowShouldClose(window)){

        glfwPollEvents();

        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();

        const ImGuiViewport *viewport = ImGui::GetMainViewport();
        ImGui::SetNextWindowPos(viewport->WorkPos);
        ImGui::SetNextWindowSize(viewport->WorkSize);

        static ImGuiDockNodeFlags dockspaceFlags = ImGuiDockNodeFlags_PassthruCentralNode;
        static ImGuiWindowFlags windowFlags = ImGuiWindowFlags_MenuBar | ImGuiWindowFlags_NoDocking | ImGuiWindowFlags_NoDecoration;


        // Weak ptr because we want to destroy opengl context in GUI::~GUI()
        static std::weak_ptr<Viewport> current_viewport = viewports["Sakura Raw Output"];


        const std::string debugTitle = "Debug Info";
        const std::string mainWindowTitle = "Main Window";

        {
            ImGui::Begin("DockSpace", nullptr, windowFlags);

            ImGuiID dockspaceId = ImGui::GetID("BaseDockSpace");
            ImGui::DockSpace(dockspaceId, ImVec2(0.0f, 0.0f), dockspaceFlags);

            [[maybe_unused]] static bool initDockspace = [&]() {


                ImGui::DockBuilderRemoveNode(dockspaceId);// clear any previous layout
                ImGui::DockBuilderAddNode(dockspaceId, dockspaceFlags | ImGuiDockNodeFlags_DockSpace);
                ImGui::DockBuilderSetNodeSize(dockspaceId, viewport->Size);

                ImGuiID dbg, output;


                ImGui::DockBuilderSplitNode(dockspaceId, ImGuiDir_Left, 0.2f, &dbg, &output);
                ImGui::DockBuilderDockWindow(debugTitle.c_str(), dbg);
                ImGui::DockBuilderDockWindow(mainWindowTitle.c_str(), output);

                ImGui::DockBuilderFinish(dockspaceId);
                return true;
            }();


            ImGui::End();
        }

        {
            static float f = 0.0f;
            static int counter = 0;

            ImGui::Begin(debugTitle.c_str());

//            ImGui::Text("Progress: %f percent", scene.getPercentage());
            auto now = std::chrono::high_resolution_clock ::now();
            auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(now - start).count();
//            ImGui::Text("%f / %f ms", duration * 1.0f,  100.f * duration / scene.getPercentage());

            if(ImGui::BeginCombo("Viewport", current_viewport.lock()->getTitle().c_str(), 0)) {
                for(auto &[name, vp] : viewports) {
                    if(ImGui::Selectable(name.c_str(), current_viewport.lock()->getTitle() == name)) {
                        current_viewport = std::weak_ptr(vp);
                        std::cout << "Selecting viewport " << name << "\n";
                    }
                }
                ImGui::EndCombo();
            }

            ImGui::Text("Denoiser Enabled: %s", scene.denoiserEnabled ? "true" : "false");

            ImGui::Text("Application average %.3f ms/frame (%.1f FPS)", 1000.0f / ImGui::GetIO().Framerate, ImGui::GetIO().Framerate);
            ImGui::End();
        }

        {
            ImGui::Begin(mainWindowTitle.c_str(), nullptr, ImGuiWindowFlags_NoDecoration);

            if(ImGui::IsKeyDown(ImGuiKey_MouseRight)){
                scene.denoiserEnabled = true;
                current_viewport = viewports["Sakura Denoiser"];
            } else if(scene.denoiserEnabled) {
                scene.denoiserEnabled = false;
                current_viewport = viewports["Sakura Raw Output"];
            }


            current_viewport.lock()->handleInput();

            current_viewport.lock()->renderFrame();
            current_viewport.lock()->drawTexture();





            ImGui::End();
        }



        // Rendering
        ImGui::Render();
        int display_w, display_h;
        glfwGetFramebufferSize(window, &display_w, &display_h);
        glViewport(0, 0, display_w, display_h);
        glClear(GL_COLOR_BUFFER_BIT);
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

        if(ImGui::GetIO().ConfigFlags & ImGuiConfigFlags_ViewportsEnable) {
            GLFWwindow *backup_current_context = glfwGetCurrentContext();
            ImGui::UpdatePlatformWindows();
            ImGui::RenderPlatformWindowsDefault();
            glfwMakeContextCurrent(backup_current_context);
        }else{
            std::cerr << "ImGuiConfigFlags_ViewportsEnable not set. No multi window support.\n";
        }

        glfwSwapBuffers(window);

        if(current_viewport.lock()->isDone()){
            glfwSetWindowShouldClose(window, true);
            for(const auto &[name, vp] : viewports){
                vp->save();
            }
        }
    }
}
GUI::~GUI() {
    //Make sure that the viewports are destroyed before the opengl context is destroyed
    for(auto &[name, vp] : viewports){
        vp.reset();
    }

    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();

    glfwDestroyWindow(window);
    glfwTerminate();
}
__host__ void GUI::setViewports(const std::vector<std::shared_ptr<Viewport>> &vps) {
    for(auto &vp : vps){
        viewports[vp->getTitle()] = vp;
    }
}
