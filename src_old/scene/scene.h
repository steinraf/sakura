//
// Created by steinraf on 19/08/22.
//

#pragma once


#include <filesystem>
#include <functional>

#include "sceneLoader.h"

#include "pngwriter.h"


#include "../cudaHelpers.cuh"
#include "../utility/meshLoader.h"
#include "../utility/vector.cuh"

#include "../../src/gui/viewport.cuh"
#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"

#include <GL/glew.h>
#include <GLFW/glfw3.h>




class Scene {
public:
    __host__ explicit Scene(SceneRepresentation &&sceneRepr);


    __host__ ImVec2 getDimensions() const noexcept {
        return {static_cast<float>(sceneRepresentation.sceneInfo.width),
                static_cast<float>(sceneRepresentation.sceneInfo.height)};
    }


    bool denoiserEnabled = false;

    TLAS const *getMeshAccelerationStructure();

    Camera deviceCamera;
    SceneRepresentation sceneRepresentation;


private:

    std::vector<thrust::device_vector<Triangle>> hostDeviceMeshTriangleVec;
    std::vector<thrust::device_vector<float>> hostDeviceMeshCDF;
    std::vector<float> totalMeshArea;

    std::vector<thrust::device_vector<Triangle>> hostDeviceEmitterTriangleVec;
    std::vector<thrust::device_vector<float>> hostDeviceEmitterCDF;
    std::vector<float> totalEmitterArea;

    TLAS *meshAccelerationStructure{};

};
