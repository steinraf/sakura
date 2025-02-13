//
// Created by steinraf on 04.02.25.
//

#pragma once

#include <Eigen/Dense>

#include "../geometry/ray.cuh"
#include "../rng/sampler.cuh"

class Camera {
public:
    __host__ __device__ Camera() noexcept;
    __host__ __device__ Camera(Eigen::Isometry3f tf, float fov,
                               float aspectRatio, float aperture,
                               float focusDist, float near = 0.1f,
                               float far = 100.0f) noexcept;

    // Takes in screen-space coordinates u, v and sampler
    // returns ray originating from the camera
    __device__ Ray getRay(float u, float v, Sampler& sampler) const;

    __host__ void createTranslationSlider();

    __host__ __device__ void translate(const Eigen::Vector3f& x);

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
    CameraBuilder& setTransform(Eigen::Isometry3f tf);
    CameraBuilder& setFOV(float fov);
    CameraBuilder& setAspectRatio(float aspectRatio);
    CameraBuilder& setAperture(float aperture);
    CameraBuilder& setFocusDist(float focusDist);
    CameraBuilder& setNear(float near);
    CameraBuilder& setFar(float far);
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
