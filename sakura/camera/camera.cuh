//
// Created by steinraf on 04.02.25.
//

#pragma once

#include "../common.cuh"
#include "../geometry/ray.cuh"
#include <Eigen/Dense>


class Camera {
public:
    CPU_GPU Camera() noexcept;
    CPU_GPU Camera(Eigen::Isometry3f tf, float fov,
                   float aspectRatio, float aperture,
                   float focusDist, float near = 0.1f,
                   float far = 10000.0f) noexcept;

    CPU_GPU static Eigen::Isometry3f lookAt(const Vec3f &center, const Vec3f &lookAt, const Vec3f &up);

    // Takes in screen-space coordinates u, v and sampler
    // returns ray originating from the camera
    __device__ Ray getRay(float u, float v, Sampler &sampler) const;

    __host__ void createTranslationSlider();

    CPU_GPU void translate(const Vec3f &x);
    CPU_GPU void relativeTranslate(const Vec3f &x);

    CPU_GPU void updateFOV(float fov);
    CPU_GPU void updateLensRadius(float aperture);

    float focusDist;

    CPU_GPU void setFocusPlane(const Vec3f &point);

private:
    Eigen::Isometry3f cameraTransform;
    Eigen::Projective3f sampleToCamera;
    float k;
    float near, far;
    float lensRadius;
    float aspectRatio;
    CPU_GPU void generateSampleToCameraMatrix();
    CPU_GPU void setK(float fov);
};

class CameraBuilder {
public:
    CameraBuilder() = default;
    CameraBuilder &setTransform(Eigen::Isometry3f transform);
    CameraBuilder &setFOV(float fieldOfView);
    CameraBuilder &setAspectRatio(float ratio);
    CameraBuilder &setAperture(float apertureSize);
    CameraBuilder &setFocusDist(float dist);
    CameraBuilder &setNear(float nearDistance);
    CameraBuilder &setFar(float farDistance);
    Camera build();

private:
    Eigen::Isometry3f tf = Eigen::Isometry3f::Identity();
    float fov = 45.f;
    float aspectRatio = 16.f / 9.f;
    float aperture = 0.001;
    float focusDist = 1.0;
    float near = 0.01;
    float far = 10000.0;
};

enum class FilmFormat {
    PNG,
};

enum class ColorSpace {
    RGB,
};

struct Film {
    Eigen::Vector2<unsigned> size = {1024, 1024};
    FilmFormat format = FilmFormat::PNG;
    ColorSpace colorSpace = ColorSpace::RGB;
};

enum class SamplingStrategy {
    UNIFORM,
};

struct SamplingPattern {
    int spp = 1;
    SamplingStrategy strategy = SamplingStrategy::UNIFORM;
};

enum class ReconstructionType {
    NONE,
};

struct ReconstructionFilter {
    ReconstructionType type = ReconstructionType::NONE;
};

struct Sensor {
    Camera camera{};
    Film film{};
    ReconstructionFilter reconstructionFilter{};
    SamplingPattern samplingPattern{};
};
