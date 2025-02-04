//
// Created by steinraf on 04.02.25.
//


#include <utility>

#include "camera.cuh"

__host__ __device__ Camera::Camera() noexcept
    : Camera(Eigen::Isometry3f::Identity(), 90.f, 16.f / 9.f, 0.0f, 1.0f) {}

__host__ __device__ Camera::Camera(Eigen::Isometry3f tf, float fov,
                                   float aspectRatio, float aperture,
                                   float focusDist, float near,
                                   float far) noexcept
    : cameraTransform(std::move(tf)),
      sampleToCamera(),
      k(tanf(fov * M_PIf / 360.f)),
      near(near),
      far(far),
      focusDist(focusDist),
      lensRadius(sqrtf(2.f) / 2.f * aperture * aspectRatio) {
    sampleToCamera.matrix() << 2 * k, 0.f, 0.f, -k,       //
            0.f, -2 * k / aspectRatio, 0.f, k / aspectRatio,  //
            0.f, 0.f, 0.f, 1.f,                               //
            0.f, 0.f, (near - far) / (near * far), 1.f / near;
}
__device__ Ray Camera::getRay(float u, float v) const {
    Eigen::Vector4f nearSample =
            (sampleToCamera * Eigen::Vector4f{u, v, 0.0, 1.0});

    assert(nearSample[3] != 0.0f);

    Eigen::Vector3f nearP = Eigen::Vector3f{nearSample[0] / nearSample[3],
                                            nearSample[1] / nearSample[3],
                                            nearSample[2] / nearSample[3]};


    float ft = focusDist / nearP[2];
    Eigen::Vector3f pFocus = ft * nearP;
    Eigen::Vector3f dNorm = pFocus.normalized();

    return Ray{cameraTransform * Eigen::Vector3f{0.0, 0.0, 0.0},
               (cameraTransform.linear() * dNorm).normalized(), near, far};
}

CameraBuilder& CameraBuilder::setTransform(Eigen::Isometry3f tf) {
    this->tf = std::move(tf);
    return *this;
}

CameraBuilder& CameraBuilder::setFOV(float fov) {
    this->fov = fov;
    return *this;
}

CameraBuilder& CameraBuilder::setAspectRatio(float aspectRatio) {
    this->aspectRatio = aspectRatio;
    return *this;
}

CameraBuilder& CameraBuilder::setAperture(float aperture) {
    this->aperture = aperture;
    return *this;
}

CameraBuilder& CameraBuilder::setFocusDist(float focusDist) {
    this->focusDist = focusDist;
    return *this;
}

CameraBuilder& CameraBuilder::setNear(float near) {
    this->near = near;
    return *this;
}

CameraBuilder& CameraBuilder::setFar(float far) {
    this->far = far;
    return *this;
}

Camera CameraBuilder::build() {
    return Camera{tf, fov, aspectRatio, aperture, focusDist, near, far};
}
