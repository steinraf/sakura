//
// Created by steinraf on 04.02.25.
//

#pragma once

#include "../common.cuh"
#include "../geometry/ray.cuh"
#include <Eigen/Dense>


class Camera {
public:
    __host__ __device__ Camera() noexcept;
    __host__ __device__ Camera(Eigen::Isometry3f tf, float fov,
                               float aspectRatio, float aperture,
                               float focusDist, float near = 0.1f,
                               float far = 100.0f) noexcept;

    // Takes in screen-space coordinates u, v and sampler
    // returns ray originating from the camera
    __device__ Ray getRay(float u, float v, Sampler &sampler) const;

    __host__ void createTranslationSlider();

    __host__ __device__ void translate(const Eigen::Vector3f &x);
    __host__ __device__ void relativeTranslate(const Eigen::Vector3f &x);

private:
    Eigen::Isometry3f cameraTransform;
    Eigen::Projective3f sampleToCamera;
    float k;
    float near, far;
    float focusDist;
    float lensRadius;
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
    float aperture = 0.0;
    float focusDist = 1.0;
    float near = 0.1;
    float far = 100.0;
};
