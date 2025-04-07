//
// Created by steinraf on 07.04.25.
//

#include "viewport.cuh"

#include "../../src_old/cudaHelpers.cuh"
#include "../../src_old/scene/scene.h"
#include "../common.h"
#include "cuda_runtime.h"

void Viewport::renderFrame(bool synchronize) {
    render();
    if(synchronize) checkCudaErrors(cudaDeviceSynchronize());
}
void OpenGLViewport::render() {
    scene.step(1.f/CustomRenderer::min(ImGui::GetIO().Framerate, 1000.f));
    //TODO add correct tonemapping for live preview
    if(!scene.render()){

        is_rendering_done = true;
    }
    const auto availableSize = ImVec2{
            ImGui::GetWindowContentRegionMax().x - ImGui::GetWindowContentRegionMin().x,
            ImGui::GetWindowContentRegionMax().y - ImGui::GetWindowContentRegionMin().y,
    };
    ImGui::Image(scene.hostImageTexture, availableSize);
}
OpenGLViewport::OpenGLViewport(Scene &scene, unsigned int width, unsigned int height, std::string title)
    : scene(scene), width(width), height(height), title(std::move(title)) {

}
std::string OpenGLViewport::getTitle() const {
    return title;
}
OpenGLViewport::~OpenGLViewport() = default;
bool OpenGLViewport::isDone() const {
    return is_rendering_done;
}
void OpenGLViewport::save() {
    scene.denoise();
    scene.saveOutput();
}
