//
// Created by steinraf on 02.02.25.
//

#include <iostream>
#include <utility>

#include "gui.cuh"

#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"

#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <cuda_gl_interop.h>


static void glfw_error_callback(int error, const char* description){
    fprintf(stderr, "Glfw Error %d: %s\n", error, description);
}


GUI::GUI(bool fullscreen) : fullscreen(fullscreen) {
    glfwSetErrorCallback(glfw_error_callback);

    if (!glfwInit()){
        std::cerr << "Error while initializing glfw. Exiting.\n";
        exit(1);
    }

    const char* glsl_version = "#version 130";
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 0);

    window = [&]() -> GLFWwindow * {
        if(fullscreen) {
            const auto monitor = glfwGetPrimaryMonitor();
            const auto mode = glfwGetVideoMode(monitor);
            return glfwCreateWindow(mode->width, mode->height, applicationTitle.c_str(), monitor, nullptr);
        }else{
            int width, height;
            glfwGetMonitorWorkarea(glfwGetPrimaryMonitor(), nullptr, nullptr, &width, &height);
            return glfwCreateWindow(width, height, applicationTitle.c_str(), nullptr, nullptr);
        }
    }();

    if (window == nullptr){
        std::cerr << "Window creation failed. Exiting.\n";
        exit(1);
    }
    glfwMakeContextCurrent(window);
    glfwSwapInterval(1);

    //ImGui Setup
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO(); (void)io;
    io.ConfigFlags |= ImGuiConfigFlags_DockingEnable;
    io.ConfigFlags |= ImGuiConfigFlags_ViewportsEnable;
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableGamepad;

    ImGui::StyleColorsDark();

    ImGuiStyle& style = ImGui::GetStyle();
    if (io.ConfigFlags & ImGuiConfigFlags_ViewportsEnable){
        style.WindowRounding = 0.0f;
        style.Colors[ImGuiCol_WindowBg].w = 1.0f;
    }

    ImGui_ImplGlfw_InitForOpenGL(window, true);
    ImGui_ImplOpenGL3_Init(glsl_version);

    if(glewInit() != GLEW_OK){
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

void GUI::loop(const Scene& scene) {

    std::cout << "Initializing GUI loop\n";

    auto camTf = Eigen::Isometry3f::Identity();

    float eyeWidth = 1.0;

    camTf.translate(Eigen::Vector3f{-eyeWidth/2.0f, 5, -30});
    auto openglViewport = OpenGLViewport(1080, 1080, "viewport", Camera{camTf, 35, 1.0f, 0.0, 30.0});
    camTf.translate(Eigen::Vector3f{eyeWidth, 0, 0});
    auto openglViewport2 = OpenGLViewport(1080, 1080, "viewport2", Camera{camTf, 35, 1.0f, 0.0, 30.0});



    std::cout << "Starting GUI loop\n";

    float t = 0.0;

    while (!glfwWindowShouldClose(window)){

        t += 0.01;

        glfwPollEvents();

        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();

        const ImGuiViewport* viewport = ImGui::GetMainViewport();
        ImGui::SetNextWindowPos(viewport->WorkPos);
        ImGui::SetNextWindowSize(viewport->WorkSize);

        {
            ImGui::Begin("main_window", nullptr, ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoSavedSettings | ImGuiWindowFlags_NoBringToFrontOnFocus);

            openglViewport.renderFrame(scene);
            openglViewport2.renderFrame(scene);

            openglViewport.translateCamera(Eigen::Vector3f{std::sin(t) * 0.01f, 0, std::cos(t) * 0.01f});
            openglViewport2.translateCamera(Eigen::Vector3f{std::sin(t) * 0.01f, 0, std::cos(t) * 0.01f});



            ImGui::End();
        }

        ImGui::Render();
        int display_w, display_h;
        glfwGetFramebufferSize(window, &display_w, &display_h);
        glViewport(0, 0, display_w, display_h);
        glClear(GL_COLOR_BUFFER_BIT);
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

        if (ImGui::GetIO().ConfigFlags & ImGuiConfigFlags_ViewportsEnable)
        {
            GLFWwindow* backup_current_context = glfwGetCurrentContext();
            ImGui::UpdatePlatformWindows();
            ImGui::RenderPlatformWindowsDefault();
            glfwMakeContextCurrent(backup_current_context);
        }

        glfwSwapBuffers(window);
    }

    std::cout << "Exiting GUI loop\n";
}

Eigen::Vector2f GUI::getWindowSize() const {
    if(fullscreen){
        const auto monitor = glfwGetPrimaryMonitor();
        const auto mode = glfwGetVideoMode(monitor);
        return {mode->width, mode->height};
    }

    int width, height;
    glfwGetWindowSize(window, &width, &height);
    return {width, height};
}

OpenGLViewport::OpenGLViewport(int width, int height, std::string title, Camera camera)
        : size({static_cast<float>(width), static_cast<float>(height)}), title(std::move(title)),
        texture(0), resource(nullptr), camera(std::move(camera)) {

    glGenTextures(1, &texture);

    glBindTexture(GL_TEXTURE_2D, texture);

    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);


    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, width, height, 0, GL_RGB, GL_FLOAT, nullptr);

    checkCudaErrors(cudaGraphicsGLRegisterImage(&resource, texture, GL_TEXTURE_2D, cudaGraphicsRegisterFlagsWriteDiscard));
}

void OpenGLViewport::renderFrame(const Scene& scene) {

    ImGui::Begin(title.c_str(), nullptr, ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_NoSavedSettings);
    ImGui::SetWindowSize(size);

    //TODO map multiple resources at once to batch all viewports
    checkCudaErrors(cudaGraphicsMapResources(1, &resource, nullptr));

    cudaArray_t cudaArray;
    checkCudaErrors(cudaGraphicsSubResourceGetMappedArray(&cudaArray, resource, 0, 0));

    cudaResourceDesc resDesc = {};
    resDesc.resType = cudaResourceTypeArray;
    resDesc.res.array.array = cudaArray;

    cudaSurfaceObject_t surface;
    checkCudaErrors(cudaCreateSurfaceObject(&surface, &resDesc));

    ImGui::SliderFloat("Camera X", &camera.cameraTransform.translation()[0], -100, 100);
    ImGui::SliderFloat("Camera Y", &camera.cameraTransform.translation()[1], -100, 100);
    ImGui::SliderFloat("Camera Z", &camera.cameraTransform.translation()[2], -100, 100);


    scene.render(surface, camera, size);


    checkCudaErrors(cudaDestroySurfaceObject(surface));
    checkCudaErrors(cudaGraphicsUnmapResources(1, &resource, nullptr));


    glClear(GL_COLOR_BUFFER_BIT);
    glBindTexture(GL_TEXTURE_2D, texture);
    glBegin(GL_QUADS);
    glTexCoord2f(0, 0); glVertex2f(-1, -1);
    glTexCoord2f(1, 0); glVertex2f(1, -1);
    glTexCoord2f(1, 1); glVertex2f(1, 1);
    glTexCoord2f(0, 1); glVertex2f(-1, 1);
    glEnd();

    ImGui::Image(texture, size);

    ImGui::End();
}
void OpenGLViewport::translateCamera(const Eigen::Vector3f &translation) {
    camera.cameraTransform.translate(translation);
}
