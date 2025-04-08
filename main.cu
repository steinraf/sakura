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

    GUI gui({
            .applicationTitle = "Sakura",
            .fullscreen = false,
    });
    auto scene = Scene{SceneRepresentation(filePath)};

    bool needsRender = true;


    gui.loop(scene);



    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    std::cout << "Total time: " << duration.count() << "ms\n";

    return EXIT_SUCCESS;
}
