#include <iostream>

#include "sakura/gui/gui.cuh"


int main(int argc, char **argv){

    auto scene = SceneBuilder()
                        .addObj("scenes/models/Mesh000.obj")
                         .addObj("scenes/models/Mesh001.obj")
                         .addObj("scenes/models/Mesh002.obj")
                         .addObj("scenes/models/Mesh003.obj")
                         .addObj("scenes/models/Mesh004.obj")
                         .addObj("scenes/models/Mesh005.obj")
                         .addObj("scenes/models/Mesh006.obj")
                         .addObj("scenes/models/Mesh007.obj")
                         .addObj("scenes/models/Mesh008.obj")
                         .addObj("scenes/models/Mesh009.obj")
                         .addObj("scenes/models/Mesh010.obj")
                         .addObj("scenes/models/Mesh011.obj")
                         .addObj("scenes/models/Mesh012.obj")
                         .addObj("scenes/models/Mesh013.obj")
                         .addObj("scenes/models/Mesh014.obj")
                         .addObj("scenes/models/Mesh015.obj")
                        .build();

    auto gui = GUI(false);
    gui.loop(scene);

    return EXIT_SUCCESS;
}


//#include "imgui.h"
//#include "imgui_impl_glfw.h"
//#include "imgui_impl_opengl3.h"
//#include <imgui_internal.h>
//#include "GL/glew.h"
//#include "GLFW/glfw3.h"
//#include <iostream>
//#include <string>
//
//static void glfw_error_callback(int error, const char* description){
//    fprintf(stderr, "Glfw Error %d: %s\n", error, description);
//}
//
//int main(){
//
//    GLFWwindow *window;
//    bool fullscreen = false;
//    const std::string applicationTitle = "Sakura";
//
//
//    glfwSetErrorCallback(glfw_error_callback);
//
//    if (!glfwInit()){
//        std::cerr << "Error while initializing glfw. Exiting.\n";
//        exit(1);
//    }
//
//    const char* glsl_version = "#version 130";
//    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
//    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 0);
//
//    window = [&]() -> GLFWwindow * {
//        if(fullscreen) {
//            const auto monitor = glfwGetPrimaryMonitor();
//            const auto mode = glfwGetVideoMode(monitor);
//            return glfwCreateWindow(mode->width, mode->height, applicationTitle.c_str(), monitor, nullptr);
//        }else{
//            int width, height;
//            glfwGetMonitorWorkarea(glfwGetPrimaryMonitor(), nullptr, nullptr, &width, &height);
//            return glfwCreateWindow(width, height, applicationTitle.c_str(), nullptr, nullptr);
//        }
//    }();
//
//    if (window == nullptr){
//        std::cerr << "Window creation failed. Exiting.\n";
//        exit(1);
//    }
//    glfwMakeContextCurrent(window);
//    glfwSwapInterval(1);
//
//    //ImGui Setup
//    IMGUI_CHECKVERSION();
//    ImGui::CreateContext();
//    ImGuiIO& io = ImGui::GetIO(); (void)io;
//    io.ConfigFlags |= ImGuiConfigFlags_DockingEnable;
//    io.ConfigFlags |= ImGuiConfigFlags_ViewportsEnable;
//    io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;
//    io.ConfigFlags |= ImGuiConfigFlags_NavEnableGamepad;
//
//    ImGui::StyleColorsDark();
//
//    ImGuiStyle& style = ImGui::GetStyle();
//    if (io.ConfigFlags & ImGuiConfigFlags_ViewportsEnable){
//        style.WindowRounding = 0.0f;
//        style.Colors[ImGuiCol_WindowBg].w = 1.0f;
//    }
//
//    ImGui_ImplGlfw_InitForOpenGL(window, true);
//    ImGui_ImplOpenGL3_Init(glsl_version);
//
//    if(glewInit() != GLEW_OK){
//        std::cerr << "Glew not correctly initialized. Exiting.\n";
//        exit(1);
//    }
//
//
//    while (!glfwWindowShouldClose(window)) {
//
//        glfwPollEvents();
//
//        ImGui_ImplOpenGL3_NewFrame();
//        ImGui_ImplGlfw_NewFrame();
//        ImGui::NewFrame();
//
//        static ImGuiDockNodeFlags dockspace_flags = ImGuiDockNodeFlags_PassthruCentralNode;
//
//        // We are using the ImGuiWindowFlags_NoDocking flag to make the parent window not dockable into,
//        // because it would be confusing to have two docking targets within each others.
//        ImGuiWindowFlags window_flags = ImGuiWindowFlags_MenuBar | ImGuiWindowFlags_NoDocking;
//
//        ImGuiViewport *viewport = ImGui::GetMainViewport();
//        ImGui::SetNextWindowPos(viewport->Pos);
//        ImGui::SetNextWindowSize(viewport->Size);
//        ImGui::SetNextWindowViewport(viewport->ID);
//        ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding, 0.0f);
//        ImGui::PushStyleVar(ImGuiStyleVar_WindowBorderSize, 0.0f);
//        window_flags |= ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoCollapse | ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoMove;
//        window_flags |= ImGuiWindowFlags_NoBringToFrontOnFocus | ImGuiWindowFlags_NoNavFocus;
//
//
//        // When using ImGuiDockNodeFlags_PassthruCentralNode, DockSpace() will render our background and handle the pass-thru hole, so we ask Begin() to not render a background.
//        if(dockspace_flags & ImGuiDockNodeFlags_PassthruCentralNode)
//            window_flags |= ImGuiWindowFlags_NoBackground;
//
//        // Important: note that we proceed even if Begin() returns false (aka window is collapsed).
//        // This is because we want to keep our DockSpace() active. If a DockSpace() is inactive,
//        // all active windows docked into it will lose their parent and become undocked.
//        // We cannot preserve the docking relationship between an active window and an inactive docking, otherwise
//        // any change of dockspace/settings would lead to windows being stuck in limbo and never being visible.
//        ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0.0f, 0.0f));
//        ImGui::Begin("DockSpace", nullptr, window_flags);
//        ImGui::PopStyleVar();
//        ImGui::PopStyleVar(2);
//
//        // DockSpace
//        if(io.ConfigFlags & ImGuiConfigFlags_DockingEnable) {
//            ImGuiID dockspace_id = ImGui::GetID("MyDockSpace");
//            ImGui::DockSpace(dockspace_id, ImVec2(0.0f, 0.0f), dockspace_flags);
//
//            static auto first_time = true;
//            if(first_time) {
//                first_time = false;
//
//                ImGui::DockBuilderRemoveNode(dockspace_id);// clear any previous layout
//                ImGui::DockBuilderAddNode(dockspace_id, dockspace_flags | ImGuiDockNodeFlags_DockSpace);
//                ImGui::DockBuilderSetNodeSize(dockspace_id, viewport->Size);
//
//                // split the dockspace into 2 nodes -- DockBuilderSplitNode takes in the following args in the following order
//                //   window ID to split, direction, fraction (between 0 and 1), the final two setting let's us choose which id we want (which ever one we DON'T set as NULL, will be returned by the function)
//                //                                                              out_id_at_dir is the id of the node in the direction we specified earlier, out_id_at_opposite_dir is in the opposite direction
//                auto dock_id_left = ImGui::DockBuilderSplitNode(dockspace_id, ImGuiDir_Left, 0.2f, nullptr, &dockspace_id);
//                auto dock_id_down = ImGui::DockBuilderSplitNode(dockspace_id, ImGuiDir_Down, 0.25f, nullptr, &dockspace_id);
//
//                // we now dock our windows into the docking node we made above
//                ImGui::DockBuilderDockWindow("Down", dock_id_down);
//                ImGui::DockBuilderDockWindow("Left", dock_id_left);
//                ImGui::DockBuilderFinish(dockspace_id);
//            }
//        }
//
//        ImGui::End();
//
//        ImGui::Begin("Left");
//        ImGui::Text("Hello, left!");
//        ImGui::End();
//
//        ImGui::Begin("Down");
//        ImGui::Text("Hello, down!");
//        ImGui::End();
//
//        ImGui::Render();
//        int display_w, display_h;
//        glfwGetFramebufferSize(window, &display_w, &display_h);
//        glViewport(0, 0, display_w, display_h);
//        glClear(GL_COLOR_BUFFER_BIT);
//        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
//
//        if (ImGui::GetIO().ConfigFlags & ImGuiConfigFlags_ViewportsEnable)
//        {
//            GLFWwindow* backup_current_context = glfwGetCurrentContext();
//            ImGui::UpdatePlatformWindows();
//            ImGui::RenderPlatformWindowsDefault();
//            glfwMakeContextCurrent(backup_current_context);
//        }
//
//        glfwSwapBuffers(window);
//    }
//}