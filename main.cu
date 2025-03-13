#include <iostream>

#include "sakura/gui/gui.cuh"
#include "sakura/scene/scene.cuh"


int main(int argc, char **argv) {

    std::vector<Sensor> sensors;

    // Scenes taken from https://benedikt-bitterli.me/resources/
    auto scene = SceneBuilder()
                         .parseXML(argc == 2 ? argv[1] : "scenes/cornell-box/scene_v3.xml")
                         //                         .parseXML("scenes/dining-room/scene_v3.xml")
                         //                         .parseXML("scenes/lamp/scene_v3.xml")
                         //                         .parseXML("scenes/coffee/scene_v3.xml")
                         //                         .parseXML("scenes/living-room/scene_v3.xml")
                         //                         .parseXML("scenes/material-testball/scene_v0.6.xml")
                         //                         .parseXML("scenes/house/scene_v3.xml")
                         //                         .parseXML("scenes/car/scene_v3.xml")
                         //                         .parseXML("scenes/car2/scene_v3.xml")
                         //                         .parseXML("scenes/lego/scene_v3.xml")
                         //                         .parseXML("scenes/teapot/scene_v3.xml")
                         //                         .parseXML("scenes/kitchen/scene_v3.xml")
                         //                         .parseXML("scenes/cbox.xml")
                         //                         .parseXML("scenes/cornell-box/scene_v3.xml")
                         //                         .parseXML("scenes/dragon/scene_v3.xml")
                         .getSensors(sensors)
                         .build();

    //    sensors[0].film.size = Eigen::Vector2<unsigned>{2560, 1440};
    sensors[0].camera.focusDist = 1.f;
    //    sensors[0].camera.updateLensRadius(0.6f);
    //    return EXIT_SUCCESS;

    auto gui = GUI(false);
    gui.loop(scene, sensors);

    return EXIT_SUCCESS;
}