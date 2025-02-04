#include <iostream>

#include "sakura/gui/gui.cuh"


int main(int argc, char **argv){

    auto gui = GUI(false);
    gui.loop();

    return EXIT_SUCCESS;
}

