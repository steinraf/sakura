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

#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"

#include <GL/glew.h>
#include <GLFW/glfw3.h>


struct OpenGLSharedBuffer{
    GLuint texture{};
    cudaGraphicsResource_t resource{};
    FeatureBuffer *featureBuffer{};
    cudaSurfaceObject_t surface{};
    size_t width, height;

    CPU_ONLY explicit OpenGLSharedBuffer(size_t width, size_t height);
    CPU_ONLY ~OpenGLSharedBuffer();
};

class Scene {
public:
    __host__ explicit Scene(SceneRepresentation &&sceneRepr);

    [[nodiscard]] __host__ bool render();

    __host__ void denoise();

    __host__ void saveOutput();

    __host__ void reset() noexcept;

    __host__ void step(float dt) noexcept;

    __host__ void setCameraVelocity(const Vector3f &vel) noexcept {
        cameraVelocity = vel;
    }

    __host__ ImVec2 getDimensions() const noexcept {
        return {static_cast<float>(sceneRepresentation.sceneInfo.width),
                static_cast<float>(sceneRepresentation.sceneInfo.height)};
    }

    __host__ float getPercentage() const noexcept {
        return 100.f * actualSamples / sceneRepresentation.sceneInfo.samplePerPixel;
    }

    bool denoiserEnabled = false;


private:

    SceneRepresentation sceneRepresentation;

    const unsigned int blockSizeX = 4, blockSizeY = 4;

    const dim3 threadSize{blockSizeX, blockSizeY};

    const dim3 blockSize;





    OpenGLSharedBuffer imageBuffer;
    OpenGLSharedBuffer imageBufferDenoised;
    float *denoiseWeights;
    Vec3f *denoiseOutput;


    std::vector<thrust::device_vector<Triangle>> hostDeviceMeshTriangleVec;
    std::vector<thrust::device_vector<float>> hostDeviceMeshCDF;
    std::vector<float> totalMeshArea;

    std::vector<thrust::device_vector<Triangle>> hostDeviceEmitterTriangleVec;
    std::vector<thrust::device_vector<float>> hostDeviceEmitterCDF;
    std::vector<float> totalEmitterArea;

    Camera deviceCamera;

    Vector3f cameraVelocity{0.f};


    TLAS *meshAccelerationStructure{};

    curandState *deviceCurandState{};

    int actualSamples = 0;


public:

};
