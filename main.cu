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

