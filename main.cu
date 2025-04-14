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

//    Camera camera;
//    int spp;
//    int maxRayDepth;
//    curandState *curandState;

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

        gui.setViewports({rawOutput,
                          std::make_shared<Denoiser>(
                                  rawOutput,
                                  "Sakura Denoiser")});
    }


    gui.loop(scene);

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    std::cout << "Total time: " << duration.count() << "ms\n";



    return EXIT_SUCCESS;
}
