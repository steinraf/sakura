//
// Created by steinraf on 02.02.25.
//

#include <iostream>
#include <memory>
#include <ranges>

#include "../denoise/denoise.cuh"
#include "../scene/scene.cuh"
#include "gui.cuh"

#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"
#include "imgui_internal.h"

#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <cuda_gl_interop.h>
#include <thread>

#include "viewport.cuh"


static void glfw_error_callback(int error, const char *description) {
    fprintf(stderr, "Glfw Error %d: %s\n", error, description);
}


GUI::GUI(bool fullscreen) : fullscreen(fullscreen) {
    glfwSetErrorCallback(glfw_error_callback);

    if(!glfwInit()) {
        std::cerr << "Error while initializing glfw. Exiting.\n";
        exit(1);
    }

    const char *glsl_version = "#version 130";
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 0);

    window = [&]() -> GLFWwindow * {
        if(fullscreen) {
            const auto monitor = glfwGetPrimaryMonitor();
            const auto mode = glfwGetVideoMode(monitor);
            return glfwCreateWindow(mode->width, mode->height, applicationTitle.c_str(), monitor, nullptr);
        } else {
            int width, height;
            glfwGetMonitorWorkarea(glfwGetPrimaryMonitor(), nullptr, nullptr, &width, &height);
            return glfwCreateWindow(width, height, applicationTitle.c_str(), nullptr, nullptr);
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

GUI::~GUI() {
    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();

    glfwDestroyWindow(window);
    glfwTerminate();
}

struct Functor {
    __device__ Eigen::Vector3f operator()(const Eigen::Vector3f &v) const {
        auto tf = [] __device__(float f) {
            return 0.5f * (1.0f + std::tanh(f));
        };

        return Eigen::Vector3f{
                tf(v[0]),
                tf(v[1]),
                tf(v[2]),
        };
    }
};

struct FunctorPositive {
    __device__ Color operator()(const Eigen::Vector3f &v) const {
        auto tf = [] __device__(float f) {
            return 0.5f * (1.0f + std::tanh(2.f * f - 1.f));
        };

        return Color{
                tf(v[0]),
                tf(v[1]),
                tf(v[2]),
        };
    }
};

struct FunctorIdentity {
    __device__ Color operator()(const Eigen::Vector3f &v) const {
        return v;
    }
};

struct FunctorBounded {
    const Eigen::Array3f min;
    const Eigen::Array3f extent;
    explicit FunctorBounded(const AABB &bb) : min(bb.min.array()), extent((bb.max - bb.min).array()) {}
    __device__ Color operator()(const Eigen::Vector3f &vec) const {
        return (vec.array() - min) / extent;
    }
};

struct FunctorUV {
    __device__ Color operator()(const Eigen::Vector3f &v) const {
        const int numChecker = 4;
        const Color light = Color{0.8f, 0.8f, 0.8f};
        const Color dark = Color{0.2f, 0.2f, 0.2f};

        const auto checker = [](float u, float v) -> bool {
            int x = std::floor(numChecker * u);
            int y = std::floor(numChecker * v);
            return (x + y) % 2 == 0;
        };

        //        Color output = checker(v[0], v[1]) ? light : dark;
        Color output = Color{v[0], v[1], 0.0f};
        return output;
    }
};

void GUI::loop(const Scene &scene, std::vector<Sensor> sensors) {

    std::cout << "Initializing GUI loop\n";

    std::vector<std::shared_ptr<Renderable>> viewports;

    if(sensors.size() != 1) {
        std::cerr << "Only 1 sensor can be loaded from a file\n";
        std::cerr << "Truncating sensors and continuing\n";
        sensors.resize(1);
    }
    //
    static constexpr float eyeWidth = 0.1;

    const auto width = sensors[0].film.size[0];
    const auto height = sensors[0].film.size[1];


    Camera cam = sensors[0].camera;

    auto vp = std::make_shared<OpenGLViewport>(scene, width, height, "Raw Output", cam);

    //    viewports.push_back(std::make_shared<Denoiser>(vp->getFeatureBuffer(), width, height, "Denoised Output"));
    viewports.push_back(vp);

    //    cam.translate(Eigen::Vector3f{eyeWidth, 0, 0});
    //    auto vpClone = std::make_shared<OpenGLViewport>(scene, width, height, "Offset Viewport", cam);
    //    viewports.emplace_back(vpClone);

    viewports.push_back(std::make_shared<BufferVisualizer<Functor, BUFFERTYPE::MEAN>>(vp->getFeatureBuffer()->normal, "Normal Buffer", width, height, Functor{}));
    //    viewports.push_back(std::make_shared<BufferVisualizer<FunctorPositive, BUFFERTYPE::SAMPLEVARIANCE>>(vp->getFeatureBuffer()->color, "Color Buffer Sample Variance", width, height, FunctorPositive{}));
    viewports.push_back(std::make_shared<BufferVisualizer<FunctorUV, BUFFERTYPE::MEAN>>(vp->getFeatureBuffer()->uv, "UV coordinates", width, height, FunctorUV{}));
    viewports.push_back(std::make_shared<BufferVisualizer<FunctorBounded, BUFFERTYPE::MEAN>>(vp->getFeatureBuffer()->position, "Position Buffer", width, height, FunctorBounded{scene.getBoundingBox()}));

    constexpr unsigned int MAX_SIZE = 4;
    assert(viewports.size() <= MAX_SIZE && "Only 4 viewports supported");
    std::cout << "Starting GUI loop\n";

    float t = 0.0;
    const float dt = 0.01;// todo account for lag
                          //
    while(!glfwWindowShouldClose(window)) {

        glfwPollEvents();

        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();

        const ImGuiViewport *viewport = ImGui::GetMainViewport();
        ImGui::SetNextWindowPos(viewport->WorkPos);
        ImGui::SetNextWindowSize(viewport->WorkSize);

        static ImGuiDockNodeFlags dockspaceFlags = ImGuiDockNodeFlags_PassthruCentralNode;
        ImGuiWindowFlags windowFlags = ImGuiWindowFlags_MenuBar | ImGuiWindowFlags_NoDocking | ImGuiWindowFlags_NoDecoration;

        {
            ImGui::Begin("DockSpace", nullptr, windowFlags);

            ImGuiIO &io = ImGui::GetIO();
            (void) io;
            io.IniFilename = nullptr;

            if(io.ConfigFlags & ImGuiConfigFlags_DockingEnable) {
                ImGuiID dockspaceId = ImGui::GetID("MyDockSpace");
                ImGui::DockSpace(dockspaceId, ImVec2(0.0f, 0.0f), dockspaceFlags);


                [[maybe_unused]] static auto createDockBuilder = [&]() -> std::vector<ImGuiID> {
                    ImGui::DockBuilderRemoveNode(dockspaceId);// clear any previous layout
                    ImGui::DockBuilderAddNode(dockspaceId, dockspaceFlags | ImGuiDockNodeFlags_DockSpace);
                    ImGui::DockBuilderSetNodeSize(dockspaceId, viewport->Size);

                    // settings and 2x2 grid of viewports (row | col)
                    ImGuiID s, a00, a10, a01, a11;

                    ImGui::DockBuilderSplitNode(dockspaceId, ImGuiDir_Left, 0.2f, &s, &a00);

                    ImGui::DockBuilderDockWindow(settingsTitle.c_str(), s);


                    if(viewports.size() == 1) {
                        ImGui::DockBuilderDockWindow(viewports[0]->getTitle().c_str(), a00);
                        ImGui::DockBuilderFinish(dockspaceId);
                        return {s, a00};
                    } else if(viewports.size() == 2) {
                        ImGui::DockBuilderSplitNode(a00, ImGuiDir_Left, 0.5f, &a00, &a01);
                        ImGui::DockBuilderDockWindow(viewports[0]->getTitle().c_str(), a00);
                        ImGui::DockBuilderDockWindow(viewports[1]->getTitle().c_str(), a01);
                        ImGui::DockBuilderFinish(dockspaceId);
                        return {s, a00, a01};
                    } else if(viewports.size() == 3) {
                        ImGui::DockBuilderSplitNode(a00, ImGuiDir_Up, 0.5f, &a00, &a10);
                        ImGui::DockBuilderSplitNode(a00, ImGuiDir_Left, 0.5f, &a00, &a01);
                        ImGui::DockBuilderDockWindow(viewports[0]->getTitle().c_str(), a00);
                        ImGui::DockBuilderDockWindow(viewports[1]->getTitle().c_str(), a01);
                        ImGui::DockBuilderDockWindow(viewports[2]->getTitle().c_str(), a10);
                        ImGui::DockBuilderFinish(dockspaceId);
                        return {s, a00, a01, a10};
                    } else if(viewports.size() == 4) {
                        ImGui::DockBuilderSplitNode(a00, ImGuiDir_Up, 0.5f, &a00, &a10);
                        ImGui::DockBuilderSplitNode(a00, ImGuiDir_Left, 0.5f, &a00, &a01);
                        ImGui::DockBuilderSplitNode(a10, ImGuiDir_Left, 0.5f, &a10, &a11);
                        ImGui::DockBuilderDockWindow(viewports[0]->getTitle().c_str(), a00);
                        ImGui::DockBuilderDockWindow(viewports[1]->getTitle().c_str(), a01);
                        ImGui::DockBuilderDockWindow(viewports[2]->getTitle().c_str(), a10);
                        ImGui::DockBuilderDockWindow(viewports[3]->getTitle().c_str(), a11);
                        ImGui::DockBuilderFinish(dockspaceId);
                        return {s, a00, a01, a10, a11};
                    } else {
                        throw std::runtime_error("Only 4 viewports supported");
                    }
                }();
            } else {
                std::cerr << "Docking not enabled\n";
            }

            ImGui::End();
        }

        auto startRender = std::chrono::high_resolution_clock::now();
        for(auto &_vp: viewports) {
            _vp->renderFrame();
        }
        checkCudaErrors(cudaDeviceSynchronize());
        auto endRender = std::chrono::high_resolution_clock::now();
        auto durationRender = std::chrono::duration<float, std::milli>(endRender - startRender).count();


        {
            ImGui::Begin(settingsTitle.c_str(), nullptr, ImGuiWindowFlags_NoDecoration);

            ImGui::Text("Scene Controls");

            if(ImGui::CollapsingHeader("Viewports", ImGuiTreeNodeFlags_DefaultOpen)) {
                ImGui::Indent(50);
                for(size_t i = 0; i < viewports.size(); ++i) {
                    ImGui::Indent(40);
                    ImGui::PushID(safe_uint_to_int(i));
                    if(ImGui::CollapsingHeader(viewports[i]->getTitle().c_str(), ImGuiTreeNodeFlags_DefaultOpen)) {
                        viewports[i]->generateSettings();
                    }
                    ImGui::PopID();
                    ImGui::Unindent(40);
                }
                ImGui::Unindent(50);
            }

            if(ImGui::CollapsingHeader("Debug Information", ImGuiTreeNodeFlags_DefaultOpen)) {
                ImGui::Indent(50);
                for(size_t i = 0; i < viewports.size(); ++i) {
                    ImGui::Indent(40);
                    ImGui::PushID(safe_uint_to_int(i + viewports.size()));
                    if(ImGui::CollapsingHeader(viewports[i]->getTitle().c_str(), ImGuiTreeNodeFlags_DefaultOpen)) {
                        viewports[i]->generateDebugInformation();
                        viewports[i]->generateTimingInformation();
                    }
                    ImGui::PopID();
                    ImGui::Unindent(40);
                }
                if(ImGui::CollapsingHeader("FPS", ImGuiTreeNodeFlags_DefaultOpen)) {
                    ImGui::Indent(40);
                    ImGui::Text("Current Frame Render Time: %f ms", durationRender);
                    ImGui::Text("FPS Target: %d achieved: %d", int(1.0 / dt), int(1000.0 / durationRender));
                    ImGui::Unindent(40);
                }


                ImGui::Unindent(50);
            }


            ImGui::End();
        }

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
        }

        glfwSwapBuffers(window);
        t += dt;

        auto diff = dt - durationRender;

        if(diff > 0) {
            std::cout << "Sleeping for " << diff << " ms\n";
            std::this_thread::sleep_for(std::chrono::milliseconds(int(diff * 1000)));
        }
    }

    std::cout << "Exiting GUI loop\n";
}

Eigen::Vector2f GUI::getWindowSize() const {
    if(fullscreen) {
        const auto monitor = glfwGetPrimaryMonitor();
        const auto mode = glfwGetVideoMode(monitor);
        return {mode->width, mode->height};
    }

    int width, height;
    glfwGetWindowSize(window, &width, &height);
    return {width, height};
}
