#include <iostream>

#include "sakura/gui/gui.cuh"
#include "sakura/scene/scene.cuh"


int main() {

    std::vector<Sensor> sensors;

    auto scene = SceneBuilder()
                         //                         .parseXML("scenes/dining-room/scene_v3.xml")
                         //                         .parseXML("scenes/cbox.xml")
                         //                         .parseXML("scenes/cornell-box/scene_v3.xml")
                         .parseXML("scenes/dragon/scene_v3.xml")
                         .getSensors(sensors)
                         .build();

    auto gui = GUI(false);
    gui.loop(scene, sensors);

    return EXIT_SUCCESS;
}