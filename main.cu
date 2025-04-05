#include <filesystem>


#include "src/utility/vector.h"
#include "src/scene/scene.h"
#include "src/scene/sceneLoader.h"


static void glfw_error_callback(int error, const char* description){
    fprintf(stderr, "Glfw Error %d: %s\n", error, description);
}

int main(int argc, char **argv){

    auto start = std::chrono::high_resolution_clock::now();

    std::cout << "Parsing obj...\n";

    if(argc != 2){
        throw std::runtime_error("Please add a file as argument.");
    }

    const std::filesystem::path filePath = argv[1];

    assert(filePath.extension() == ".xml");

    std::cout << "Starting rendering...\n";


    glfwSetErrorCallback(glfw_error_callback);
    if (!glfwInit()){
        std::cerr << "Error while initializing glfw. Exiting.\n";
        exit(1);
    }

    const char* glsl_version = "#version 130";
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 0);

    const auto monitor = glfwGetPrimaryMonitor();

    const GLFWvidmode* mode = glfwGetVideoMode(monitor);

    GLFWwindow* window = glfwCreateWindow(mode->width, mode->height, "Dear ImGui GLFW+OpenGL3 example", monitor, nullptr);
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

    ImGui::StyleColorsDark();

    ImGui_ImplGlfw_InitForOpenGL(window, true);
    ImGui_ImplOpenGL3_Init(glsl_version);

    ImVec4 clear_color = ImVec4(0.45f, 0.55f, 0.60f, 1.00f);

    if(glewInit() != GLEW_OK){
        std::cerr << "Glew not correctly initialized. Exiting.\n";
        exit(1);
    }

    Scene scene(SceneRepresentation(filePath), Device::CPU);

    bool needsRender = true;


    while (!glfwWindowShouldClose(window)){

        glfwPollEvents();

        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();

        {
            static float f = 0.0f;
            static int counter = 0;

            ImGui::Begin("Debug Info");

            ImGui::Text("Progress: %f percent", scene.getPercentage());

            const Vector3f cameraPos = scene.getCameraPosition();

            ImGui::Text("Camera Position (%f, %f, %f)", cameraPos[0], cameraPos[1], cameraPos[2]);

            ImGui::ColorEdit3("clear color", (float*)&clear_color); 

            ImGui::Text("Application average %.3f ms/frame (%.1f FPS)", 1000.0f / ImGui::GetIO().Framerate, ImGui::GetIO().Framerate);
            ImGui::End();
        }

        {
            ImGui::Begin("CUDA GPU Path Tracing");

            if(needsRender){
                scene.step(1.f/CustomRenderer::min(ImGui::GetIO().Framerate, 1000.f));
                //TODO add correct tonemapping for live preview
                if(!scene.render()){
                    needsRender = false;
                    scene.denoise();
                    scene.saveOutput();
                    glfwSetWindowShouldClose(window, true);
                }
                const auto availableSize = ImVec2{
                    ImGui::GetWindowContentRegionMax().x - ImGui::GetWindowContentRegionMin().x,
                    ImGui::GetWindowContentRegionMax().y - ImGui::GetWindowContentRegionMin().y,
                };
                ImGui::Image((void *) (intptr_t)scene.hostImageTexture, availableSize);

            }else{
                const auto availableSize = ImVec2{
                        ImGui::GetWindowContentRegionMax().x - ImGui::GetWindowContentRegionMin().x,
                        ImGui::GetWindowContentRegionMax().y - ImGui::GetWindowContentRegionMin().y,
                };
                ImGui::Image((void *) (intptr_t)scene.hostImageTexture, availableSize);
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
        glClearColor(clear_color.x * clear_color.w, clear_color.y * clear_color.w, clear_color.z * clear_color.w, clear_color.w);
        glClear(GL_COLOR_BUFFER_BIT);
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

        glfwSwapBuffers(window);


    }


    // Cleanup
    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();

    glfwDestroyWindow(window);
    glfwTerminate();

    scene.saveOutput();
    std::cout << "Drew image to file\n";

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    std::cout << "Total time: " << duration.count() << "ms\n";

    return EXIT_SUCCESS;
}
