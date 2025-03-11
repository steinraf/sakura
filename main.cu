#include <iostream>

#include "sakura/gui/gui.cuh"
#include "sakura/scene/scene.cuh"


int main() {

    std::vector<Sensor> sensors;

    // Scenes taken from https://benedikt-bitterli.me/resources/
    auto scene = SceneBuilder()
                         //                         .parseXML("scenes/dining-room/scene_v3.xml")

                         //                         .parseXML("scenes/lamp/scene_v3.xml")
                         //                         .parseXML("scenes/coffee/scene_v3.xml")
                         //                         .parseXML("scenes/living-room/scene_v3.xml")
                         //                         .parseXML("scenes/material-testball/scene_v0.6.xml")
                         //                         .parseXML("scenes/house/scene_v3.xml")
                         //                         .parseXML("scenes/car/scene_v3.xml")
                         .parseXML("scenes/car2/scene_v3.xml")
                         //                         .parseXML("scenes/lego/scene_v3.xml")
                         //                         .parseXML("scenes/teapot/scene_v3.xml")
                         //                         .parseXML("scenes/kitchen/scene_v3.xml")
                         //                         .parseXML("scenes/cbox.xml")
                         //                         .parseXML("scenes/cornell-box/scene_v3.xml")
                         //                         .parseXML("scenes/dragon/scene_v3.xml")
                         .getSensors(sensors)
                         .build();

    auto gui = GUI(false);
    gui.loop(scene, sensors);

    return EXIT_SUCCESS;
}