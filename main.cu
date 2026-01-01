#include <iostream>

#include "sakura/gui/gui.cuh"
#include "sakura/scene/scene.cuh"


int main(int argc, char **argv) {


    std::vector<Sensor> sensors;

    // A lot of scenes are taken from https://benedikt-bitterli.me/resources/
    const auto scene = SceneBuilder()
                               .parseXML(argc == 2 ? argv[1] : "scenes/cornell-box/scene_v3.xml")
                               .getSensors(sensors)
                               .build();

    //    sensors[0].camera.updateLensRadius(0.1f);

    auto gui = GUI(false);
    gui.loop(scene, sensors);

    return EXIT_SUCCESS;
}