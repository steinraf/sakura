//
// Created by steinraf on 21/08/22.
//

#pragma once

#include "../common.h"

#include "../../src_old/utility/ray.h"
#include "../../src_old/utility/warp.h"


enum struct ApertureType{
    Circular,
    Triangular,
    Square,
};

#include <eigen3/Eigen/Dense>

class Camera {
public:
    CPU_GPU Camera() noexcept;
    CPU_GPU Camera(Eigen::Isometry3f tf, float fov,
                   float aspectRatio, float aperture,
                   float focusDist, float near = 0.1f,
                   float far = 10000.0f, ApertureType apertureType = ApertureType::Circular) noexcept;

    CPU_GPU Camera(Vector3f origin, Vector3f lookAt, Vector3f _up, float vFOV, float aspectRatio,
                             float aperture, float focusDist, float k1, float k2, ApertureType apertureType);

    CPU_GPU static Eigen::Isometry3f lookAt(const Vec3f &center, const Vec3f &lookAt, const Vec3f &up);

    // Takes in screen-space coordinates u, v and sample
    // returns ray originating from the camera
    __device__ Ray3f getRay(float u, float v, const Vector2f &sample) const;

    CPU_GPU void translate(const Vec3f &x);
    CPU_GPU void translateRelative(const Vec3f &x);

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
    ApertureType apertureType;
    CPU_GPU void generateSampleToCameraMatrix();
    CPU_GPU void setK(float fov);
};

