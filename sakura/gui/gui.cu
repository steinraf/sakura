//
// Created by steinraf on 02.02.25.
//

#include <iostream>
#include <ranges>
#include <utility>

#include "../denoise/denoise.cuh"
#include "../integrator/integrators.cuh"
#include "gui.cuh"

#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"
#include "imgui_internal.h"

#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <cuda_gl_interop.h>


static constexpr int RNG_SEED = 42;


static void glfw_error_callback(int error, const char *description) {
    fprintf(stderr, "Glfw Error %d: %s\n", error, description);
}

__global__ void initializeRNG(curandState *rngStates, size_t numSamples) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    if(idx >= numSamples) return;

    curand_init(RNG_SEED, idx, 0, &rngStates[idx]);
}

__global__ void renderBuffer(const Statistic<Eigen::Vector3f> *stat, cudaSurfaceObject_t surface, int width, int height, auto f) {

    size_t pixelIndex = blockIdx.x * blockDim.x + threadIdx.x;

    if(pixelIndex >= width * height) return;

    size_t x = width - 1 - pixelIndex % width, y = pixelIndex / width;

    auto color = f(stat[pixelIndex].getMean());

    uchar4 color4 = make_uchar4(color[0] * 255,
                                color[1] * 255,
                                color[2] * 255, 255);


    surf2Dwrite(color4, surface, x * sizeof(uchar4), y);
}


__global__ void clearFeatureBuffer(FeatureBuffer *buffer) {

    auto idx = blockIdx.x * blockDim.x + threadIdx.x;

    if(idx >= buffer->numElements) {
        printf("Terminating early for idx %d\n", idx);
        return;
    }

    buffer->color[idx] = {};
    buffer->normal[idx] = {};
    buffer->position[idx] = {};
    buffer->albedo[idx] = {};
}

__host__ FeatureBuffer::FeatureBuffer(size_t numElements) : numElements(numElements), color(nullptr), normal(nullptr), position(nullptr), albedo(nullptr) {
    checkCudaErrors(cudaMallocManaged(&color, numElements * sizeof(Statistic<Eigen::Vector3f>)));
    checkCudaErrors(cudaMallocManaged(&normal, numElements * sizeof(Statistic<Eigen::Vector3f>)));
    checkCudaErrors(cudaMallocManaged(&position, numElements * sizeof(Statistic<Eigen::Vector3f>)));
    checkCudaErrors(cudaMallocManaged(&albedo, numElements * sizeof(Statistic<Eigen::Vector3f>)));
    clear();
}

__host__ FeatureBuffer::~FeatureBuffer() {
    checkCudaErrors(cudaFree(color));
    checkCudaErrors(cudaFree(normal));
    checkCudaErrors(cudaFree(position));
    checkCudaErrors(cudaFree(albedo));
}
void FeatureBuffer::clear() {
    int threadsPerBlock = 256;
    size_t blocksPerGrid = (numElements + threadsPerBlock - 1) / threadsPerBlock;

    clearFeatureBuffer<<<blocksPerGrid, threadsPerBlock>>>(this);
    checkCudaErrors(cudaDeviceSynchronize());
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

void GUI::loop(const Scene &scene) {

    std::cout << "Initializing GUI loop\n";

    constexpr int MAX_SIZE = 4;// code breaks if viewports get constructed/destroyed

    std::vector<std::shared_ptr<Renderable>> viewports;
    viewports.reserve(MAX_SIZE);

    auto camTf = Eigen::Isometry3f::Identity();
    static constexpr float eyeWidth = 1.0;

    constexpr int width = 1024;
    constexpr int height = 1024;

    camTf.translate(Eigen::Vector3f{-eyeWidth / 2.0f, 5, -30});
    auto vp = std::make_shared<OpenGLViewport>(scene, width, height, "Raw Output", Camera{camTf, 35, 1.0f, 0.01, 30.0});

    viewports.push_back(std::make_shared<Denoiser>(vp->getFeatureBuffer(), width, height, "Denoised Output"));
    viewports.push_back(vp);

    viewports.push_back(std::make_shared<BufferVisualizer<Functor>>(vp->getFeatureBuffer()->normal, "Color Buffer", width, height, Functor{}));
    viewports.push_back(std::make_shared<BufferVisualizer<Functor>>(vp->getFeatureBuffer()->position, "Position Buffer", width, height, Functor{}));

    //    camTf.translate(Eigen::Vector3f{eyeWidth, 0, 0});
    //    viewports.emplace_back(width, height, "right eye", Camera{camTf, 35, 1.0f, 0.01, 30.0});


    assert(viewports.size() <= MAX_SIZE && "Only 4 viewports supported");
    std::cout << "Starting GUI loop\n";

    float t = 0.0;
    const float dt = 0.01;

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

        for(auto &_vp: viewports) {
            _vp->renderFrame();
        }


        {
            ImGui::Begin(settingsTitle.c_str(), nullptr, ImGuiWindowFlags_NoDecoration);

            ImGui::Text("Scene Controls");

            for(size_t i = 0; i < viewports.size(); ++i) {
                ImGui::Text("Viewport %s", viewports[i]->getTitle().c_str());
                ImGui::PushID(i);
                viewports[i]->generateSettings();
                ImGui::PopID();
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

OpenGLViewport::OpenGLViewport(const Scene &scene, int width, int height, std::string title, Camera camera, int spp)
    : scene(scene), size({static_cast<float>(width), static_cast<float>(height)}), title(std::move(title)),
      texture(0), resource(nullptr), featureBuffer(nullptr), surface(), camera(std::move(camera)), rngStates(nullptr), samplesPerPixel(spp) {

    checkCudaErrors(cudaMallocManaged(
            &rngStates,
            width * height * sizeof(curandState)));

    checkCudaErrors(cudaMallocManaged(
            &featureBuffer,
            sizeof(FeatureBuffer)));

    new(featureBuffer) FeatureBuffer(width * height);

    //    *featureBuffer = FeatureBuffer(width * height);

    int threadsPerBlock = 256;// Optimal number of threads per block
    int blocksPerGrid = (width * height + threadsPerBlock - 1) / threadsPerBlock;

    initializeRNG<<<blocksPerGrid, threadsPerBlock>>>(
            rngStates, width * height);

    checkCudaErrors(cudaDeviceSynchronize());

    glGenTextures(1, &texture);

    glBindTexture(GL_TEXTURE_2D, texture);

    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);


    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, width, height, 0, GL_RGB, GL_FLOAT, nullptr);


    checkCudaErrors(cudaGraphicsGLRegisterImage(&resource, texture, GL_TEXTURE_2D, cudaGraphicsRegisterFlagsNone));


    //TODO map multiple resources at once to batch all viewports
    checkCudaErrors(cudaGraphicsMapResources(1, &resource, nullptr));

    cudaArray_t cudaArray;
    checkCudaErrors(cudaGraphicsSubResourceGetMappedArray(&cudaArray, resource, 0, 0));

    cudaResourceDesc resDesc = {};
    resDesc.resType = cudaResourceTypeArray;
    resDesc.res.array.array = cudaArray;


    checkCudaErrors(cudaCreateSurfaceObject(&surface, &resDesc));
}

void OpenGLViewport::renderFrame() {

    static float t = 0.0;
    constexpr float dt = 0.01;
    t += dt;
    float circleScale = 1.0;
    translateCamera(dt * Eigen::Vector3f{circleScale * std::sin(t), 0.0, circleScale * std::cos(t)});


    ImGui::Begin(title.c_str(), nullptr, ImGuiWindowFlags_NoDecoration);// , nullptr, ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_NoSavedSettings
                                                                        //    ImGui::SetWindowSize(size);


    scene.render(surface, featureBuffer, camera, rngStates, size, samplesPerPixel);


    glClear(GL_COLOR_BUFFER_BIT);
    glBindTexture(GL_TEXTURE_2D, texture);
    glBegin(GL_QUADS);
    glTexCoord2f(0, 0);
    glVertex2f(-1, -1);
    glTexCoord2f(1, 0);
    glVertex2f(1, -1);
    glTexCoord2f(1, 1);
    glVertex2f(1, 1);
    glTexCoord2f(0, 1);
    glVertex2f(-1, 1);
    glEnd();

    ImGui::Image(texture, size);

    ImGui::End();
}
void OpenGLViewport::translateCamera(const Eigen::Vector3f &translation) {
    camera.translate(translation);
    featureBuffer->clear();
}
void OpenGLViewport::generateSettings() {
    ImGui::SliderInt("Samples per Pixel", &samplesPerPixel, 1, 16);
}
OpenGLViewport::~OpenGLViewport() {

    checkCudaErrors(cudaDestroySurfaceObject(surface));
    checkCudaErrors(cudaGraphicsUnmapResources(1, &resource, nullptr));

    checkCudaErrors(cudaFree(rngStates));
    checkCudaErrors(cudaFree(featureBuffer));

    checkCudaErrors(cudaGraphicsUnregisterResource(resource));
    glDeleteTextures(1, &texture);
}
std::string OpenGLViewport::getTitle() const {
    return title;
}
void OpenGLViewport::generateDebugInformation() {
}
Eigen::Vector2f OpenGLViewport::getWindowSize() const {
    return Eigen::Vector2f{size[0], size[1]};
}
FeatureBuffer *OpenGLViewport::getFeatureBuffer() const {
    return featureBuffer;
}


template<typename F>
BufferVisualizer<F>::BufferVisualizer(Statistic<Eigen::Vector3f> *stat, std::string title, int width, int height, F f)
    : stat(stat), size(width, height), title(std::move(title)), f(f) {


    glGenTextures(1, &texture);

    glBindTexture(GL_TEXTURE_2D, texture);

    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, width, height, 0, GL_RGB, GL_FLOAT, nullptr);


    checkCudaErrors(cudaGraphicsGLRegisterImage(&resource, texture, GL_TEXTURE_2D, cudaGraphicsRegisterFlagsNone));


    //TODO map multiple resources at once to batch all viewports
    checkCudaErrors(cudaGraphicsMapResources(1, &resource, nullptr));

    cudaArray_t cudaArray;
    checkCudaErrors(cudaGraphicsSubResourceGetMappedArray(&cudaArray, resource, 0, 0));

    cudaResourceDesc resDesc = {};
    resDesc.resType = cudaResourceTypeArray;
    resDesc.res.array.array = cudaArray;


    checkCudaErrors(cudaCreateSurfaceObject(&surface, &resDesc));
}

template<typename F>
BufferVisualizer<F>::~BufferVisualizer() {
    checkCudaErrors(cudaDestroySurfaceObject(surface));
    checkCudaErrors(cudaGraphicsUnmapResources(1, &resource, nullptr));

    checkCudaErrors(cudaGraphicsUnregisterResource(resource));
    glDeleteTextures(1, &texture);
}

template<typename F>
std::string BufferVisualizer<F>::getTitle() const {
    return title;
}

template<typename F>
void BufferVisualizer<F>::generateDebugInformation() {
}
template<typename F>
void BufferVisualizer<F>::generateSettings() {
}
template<typename F>
void BufferVisualizer<F>::renderFrame() {

    ImGui::Begin(title.c_str(), nullptr, ImGuiWindowFlags_NoDecoration);// , nullptr, ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_NoSavedSettings
                                                                        //    ImGui::SetWindowSize(size);

    int threadsPerBlock = 256;// Optimal number of threads per block
    int blocksPerGrid = (size[0] * size[1] + threadsPerBlock - 1) / threadsPerBlock;

    renderBuffer<<<blocksPerGrid, threadsPerBlock>>>(stat, surface, size[0], size[1], f);
    checkCudaErrors(cudaDeviceSynchronize());


    glClear(GL_COLOR_BUFFER_BIT);
    glBindTexture(GL_TEXTURE_2D, texture);
    glBegin(GL_QUADS);
    glTexCoord2f(0, 0);
    glVertex2f(-1, -1);
    glTexCoord2f(1, 0);
    glVertex2f(1, -1);
    glTexCoord2f(1, 1);
    glVertex2f(1, 1);
    glTexCoord2f(0, 1);
    glVertex2f(-1, 1);
    glEnd();

    ImGui::Image(texture, size);

    ImGui::End();
}


Denoiser::Denoiser(FeatureBuffer *buffer, int width, int height, std::string title)
    : buffer(buffer), size(width, height), title(std::move(title)), texture(0), resource(nullptr), surface() {

    glGenTextures(1, &texture);

    glBindTexture(GL_TEXTURE_2D, texture);

    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, width, height, 0, GL_RGB, GL_FLOAT, nullptr);


    checkCudaErrors(cudaGraphicsGLRegisterImage(&resource, texture, GL_TEXTURE_2D, cudaGraphicsRegisterFlagsNone));


    //TODO map multiple resources at once to batch all viewports
    checkCudaErrors(cudaGraphicsMapResources(1, &resource, nullptr));

    cudaArray_t cudaArray;
    checkCudaErrors(cudaGraphicsSubResourceGetMappedArray(&cudaArray, resource, 0, 0));

    cudaResourceDesc resDesc = {};
    resDesc.resType = cudaResourceTypeArray;
    resDesc.res.array.array = cudaArray;


    checkCudaErrors(cudaCreateSurfaceObject(&surface, &resDesc));
}
Denoiser::~Denoiser() {
    checkCudaErrors(cudaDestroySurfaceObject(surface));
    checkCudaErrors(cudaGraphicsUnmapResources(1, &resource, nullptr));

    checkCudaErrors(cudaGraphicsUnregisterResource(resource));
    glDeleteTextures(1, &texture);
}
std::string Denoiser::getTitle() const {
    return title;
}
void Denoiser::generateDebugInformation() {
}
void Denoiser::generateSettings() {
}
void Denoiser::renderFrame() {

    ImGui::Begin(title.c_str(), nullptr, ImGuiWindowFlags_NoDecoration);// , nullptr, ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_NoSavedSettings
                                                                        //    ImGui::SetWindowSize(size);

    int devId = 0;
    int numSMs;
    checkCudaErrors(cudaDeviceGetAttribute(&numSMs, cudaDevAttrMultiProcessorCount, devId));


    denoise<<<32 * numSMs, 256>>>(surface, buffer, size[0], size[1]);
    checkCudaErrors(cudaDeviceSynchronize());

    glClear(GL_COLOR_BUFFER_BIT);
    glBindTexture(GL_TEXTURE_2D, texture);
    glBegin(GL_QUADS);
    glTexCoord2f(0, 0);
    glVertex2f(-1, -1);
    glTexCoord2f(1, 0);
    glVertex2f(1, -1);
    glTexCoord2f(1, 1);
    glVertex2f(1, 1);
    glTexCoord2f(0, 1);
    glVertex2f(-1, 1);
    glEnd();

    ImGui::Image(texture, size);

    ImGui::End();
}
