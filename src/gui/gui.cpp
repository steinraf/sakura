//
// Created by steinraf on 05.04.25.
//

#include "gui.h"

#include "imgui.h"

#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"
#include "imgui_internal.h"
#include "viewport.cuh"

#include <GL/glew.h>
#include <cuda_gl_interop.h>
#include <thread>

#include <iostream>

static void glfw_error_callback(int error, const char* description){
    fprintf(stderr, "Glfw Error %d: %s\n", error, description);
}

GUI::GUI(const GUIConfig &config) {
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
void GUI::loop(Scene &scene) {
    bool needsRender = true;
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

        static std::unique_ptr<Viewport> renderer_viewport = std::make_unique<OpenGLViewport>(scene, scene.getDimensions().x, scene.getDimensions().y, "Sakura Main" );;



        const std::string debugTitle = "Debug Info";

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
                ImGui::DockBuilderDockWindow(renderer_viewport->getTitle().c_str(), output);

                ImGui::DockBuilderFinish(dockspaceId);
                return true;
            }();




            ImGui::End();
        }

        {
            static float f = 0.0f;
            static int counter = 0;

            ImGui::Begin(debugTitle.c_str());

            ImGui::Text("Progress: %f percent", scene.getPercentage());
            auto now = std::chrono::high_resolution_clock ::now();
            auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(now - start).count();
            ImGui::Text("%f / %f ms", duration * 1.0f,  100.f * duration / scene.getPercentage());

            const Vector3f cameraPos = scene.getCameraPosition();

            ImGui::Text("Camera Position (%f, %f, %f)", cameraPos[0], cameraPos[1], cameraPos[2]);

            ImGui::Text("Application average %.3f ms/frame (%.1f FPS)", 1000.0f / ImGui::GetIO().Framerate, ImGui::GetIO().Framerate);
            ImGui::End();
        }

        {
            ImGui::Begin(renderer_viewport->getTitle().c_str(), nullptr, ImGuiWindowFlags_NoDecoration);

            if(needsRender){
                renderer_viewport->renderFrame();

            }else{
                const auto availableSize = ImVec2{
                        ImGui::GetWindowContentRegionMax().x - ImGui::GetWindowContentRegionMin().x,
                        ImGui::GetWindowContentRegionMax().y - ImGui::GetWindowContentRegionMin().y,
                };
                ImGui::Image(scene.hostImageTexture, availableSize);
            }

            Vector3f vCamera{0.f};


            if(ImGui::IsKeyDown(ImGuiKey_W))
                vCamera[2] += 1.f;
            if(ImGui::IsKeyDown(ImGuiKey_S))
                vCamera[2] -= 1.f;

            if(ImGui::IsKeyDown(ImGuiKey_D))
                vCamera[0] += 1.f;
            if(ImGui::IsKeyDown(ImGuiKey_A))
                vCamera[0] -= 1.f;

            if(ImGui::IsKeyDown(ImGuiKey_Space))
                vCamera[1] += 1.f;
            if(ImGui::IsKeyDown(ImGuiKey_LeftShift))
                vCamera[1] -= 1.f;

            scene.setCameraVelocity(vCamera);

            if(vCamera.squaredNorm() != 0.f)
                scene.reset();
            //            scene.denoise();

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

        if(renderer_viewport->isDone()){
            glfwSetWindowShouldClose(window, true);
            renderer_viewport->save();
        }
    }
}
GUI::~GUI() {
    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();

    glfwDestroyWindow(window);
    glfwTerminate();
}
