#include <filesystem>


#include "src_old//utility/vector.cuh"
#include "src_old/scene/scene.h"
#include "src_old/scene/sceneLoader.h"

#include "src/gui/gui.cuh"


int main(int argc, char **argv){

    auto start = std::chrono::high_resolution_clock::now();

    std::cout << "Parsing obj...\n";

    if(argc != 2){
        throw std::runtime_error("Please add a file as argument.");
    }

    const std::filesystem::path filePath = argv[1];

    assert(filePath.extension() == ".xml");

    std::cout << "Starting rendering...\n";


    auto scene = Scene{SceneRepresentation(filePath)};


    GUI gui(GUIConfig{
            "Sakura",
            false,
    });

    {
        auto rawOutput = std::make_shared<OpenGLViewport>(scene,
                                                          OpenGLViewport::OpenGLConfig{
                                                                  scene.deviceCamera,
                                                                  scene.sceneRepresentation.sceneInfo.samplePerPixel,
                                                                  scene.sceneRepresentation.sceneInfo.maxRayDepth,
                                                                  int(scene.getDimensions().x),
                                                                  int(scene.getDimensions().y)},
                                                          "Sakura Raw Output");

        auto identityF = []__device__ (const Vec3f &v) { return v; };
        auto absoluteValueF = []__device__ (const Vec3f &v) { return v.absValues(); };
        auto unboundedF = []__device__ (const Vec3f &v) {
            auto tf = [] __device__(float f) {
            return 0.5f * (1.0f + std::tanh(f));
            };

            return Vec3f{
                tf(v[0]),
                tf(v[1]),
                tf(v[2]),
            };
        };

        auto unboundedPositiveF = []__device__ (const Vec3f &v) {
            auto tf = [] __device__(float f) {
                return std::tanh(f);
            };
            return Vec3f{
                tf(v[0]),
                tf(v[1]),
                tf(v[2]),
            };
        };

        gui.setViewports({rawOutput,
                          std::make_shared<Denoiser>(
                                  rawOutput,
                                  "Sakura Denoiser"),
                          std::make_shared<BufferVisualizer<BUFFERTYPE::MEAN, decltype(absoluteValueF)>>(
                                                     rawOutput,
                                                     rawOutput->getFeatureBuffer()->normal,
                                                        "Sakura Normal Buffer",
                                  absoluteValueF),
                          std::make_shared<BufferVisualizer<BUFFERTYPE::MEAN, decltype(unboundedF)>>(
                                  rawOutput,
                                  rawOutput->getFeatureBuffer()->position,
                                  "Sakura Position Buffer",
                                  unboundedF),
                          std::make_shared<BufferVisualizer<BUFFERTYPE::MEAN, decltype(unboundedPositiveF)>>(
                                  rawOutput,
                                  rawOutput->getFeatureBuffer()->albedo,
                                  "Sakura Albedo Buffer",
                                  unboundedPositiveF),
                          std::make_shared<BufferVisualizer<BUFFERTYPE::MEAN, decltype(identityF)>>(
                                  rawOutput,
                                  rawOutput->getFeatureBuffer()->uv,
                                  "Sakura UV Buffer",
                                  identityF)
                         });
    }


    gui.loop(scene);

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    std::cout << "Total time: " << duration.count() << "ms\n";



    return EXIT_SUCCESS;
}
