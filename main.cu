#include <iostream>

#include "sakura/gui/gui.cuh"
#include "sakura/scene/scene.cuh"


int main() {

    auto scene = SceneBuilder()
                         .parseXML("scenes/cbox.xml")
                         .build();

    auto gui = GUI(false);
    gui.loop(scene);

    return EXIT_SUCCESS;
}