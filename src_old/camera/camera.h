//
// Created by steinraf on 21/08/22.
//

#pragma once

#include "../utility/ray.h"
#include "../utility/vector.cuh"


class Camera {
public:
    __host__ __device__ Camera(Vector3f origin, Vector3f lookAt, Vector3f _up, float vFOV, float aspectRatio,
                               float aperture, float focusDist, float k1, float k2);

    __device__ Ray3f getRay(float s, float t, const Vector2f &sample) const;

    //relative velocity with right/up/front
    __host__ __device__ void addVelocity(const Vector3f v, float t) noexcept {
        cameraToWorld.addPosition((v[0] * right + v[1] * up + v[2] * front)* t);
    }

    __host__ __device__ Vector3f getPosition() const noexcept{
        return cameraToWorld.getPosition();
    }

private:
    Vector3f origin;
    Vector3f right, up, front;

    Matrix4f sampleToCamera;
    Matrix4f cameraToWorld;

    const float lensRadius;
    const float focusDist;
    const float k1, k2;
    const float far = 100000.f;
    const float near = 0.0001;
};
