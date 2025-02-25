//
// Created by steinraf on 04.02.25.
//

#include <utility>

#include "imgui.h"

#include "../geometry/ray.cuh"
#include "../rng/sampler.cuh"
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

    sampleToCamera.matrix() << 2 * k, 0.f, 0.f, -k,         //
            0.f, -2 * k / aspectRatio, 0.f, k / aspectRatio,//
            0.f, 0.f, 0.f, 1.f,                             //
            0.f, 0.f, (near - far) / (near * far), 1.f / near;
}
__device__ Ray Camera::getRay(float u, float v, Sampler &sampler) const {
    Eigen::Vector4f nearSample = (sampleToCamera * Eigen::Vector4f{u, v, 0.0, 1.0});

    assert(nearSample[3] != 0.0f);

    Eigen::Vector3f nearP = Eigen::Vector3f{nearSample[0] / nearSample[3],
                                            nearSample[1] / nearSample[3],
                                            nearSample[2] / nearSample[3]};

    auto diskSampleLens = sample::squareToUniformDisk(sampler.getSample2D());

    Eigen::Vector3f pLens =
            lensRadius * Eigen::Vector3f{diskSampleLens[0], diskSampleLens[1], 0.0};

    float ft = focusDist / nearP[2];
    Eigen::Vector3f pFocus = ft * nearP;
    Eigen::Vector3f dNorm = (pFocus - pLens).normalized();

    return Ray{cameraTransform * pLens,
               (cameraTransform.linear() * dNorm).normalized(), near, far};
}
__host__ void Camera::createTranslationSlider() {
    ImGui::SliderFloat("Camera X", &cameraTransform.translation()[0], -100, 100);
    ImGui::SliderFloat("Camera Y", &cameraTransform.translation()[1], -100, 100);
    ImGui::SliderFloat("Camera Z", &cameraTransform.translation()[2], -100, 100);
}
__host__ __device__ void Camera::translate(const Eigen::Vector3f &x) {
    cameraTransform.translate(x);
}
__host__ __device__ void Camera::relativeTranslate(const Eigen::Vector3f &x) {
    cameraTransform.translation() += cameraTransform.linear() * x;
}
__host__ __device__ Eigen::Isometry3f Camera::lookAt(const Eigen::Vector3f &center, const Eigen::Vector3f &lookAt, const Eigen::Vector3f &up) {
    Eigen::Vector3f f = (lookAt - center).normalized();
    Eigen::Vector3f r = up.cross(-f).normalized();
    Eigen::Vector3f u = -f.cross(r);

    Eigen::Isometry3f tf = Eigen::Isometry3f::Identity();
    tf.linear().col(0) = r;
    tf.linear().col(1) = u;
    tf.linear().col(2) = f;
    tf.translation() = center;

    printf("tf = %f, %f, %f, %f\n%f, %f, %f, %f\n%f, %f, %f, %f\n%f, %f, %f, %f\n", tf.matrix()(0, 0), tf.matrix()(0, 1), tf.matrix()(0, 2), tf.matrix()(0, 3), tf.matrix()(1, 0), tf.matrix()(1, 1), tf.matrix()(1, 2), tf.matrix()(1, 3), tf.matrix()(2, 0), tf.matrix()(2, 1), tf.matrix()(2, 2), tf.matrix()(2, 3), tf.matrix()(3, 0), tf.matrix()(3, 1), tf.matrix()(3, 2), tf.matrix()(3, 3));

    return tf;
}

CameraBuilder &CameraBuilder::setTransform(Eigen::Isometry3f transform) {
    this->tf = std::move(transform);
    return *this;
}

CameraBuilder &CameraBuilder::setFOV(float fieldOfView) {
    this->fov = fieldOfView;
    return *this;
}

CameraBuilder &CameraBuilder::setAspectRatio(float ratio) {
    this->aspectRatio = ratio;
    return *this;
}

CameraBuilder &CameraBuilder::setAperture(float apertureSize) {
    this->aperture = apertureSize;
    return *this;
}

CameraBuilder &CameraBuilder::setFocusDist(float dist) {
    this->focusDist = dist;
    return *this;
}

CameraBuilder &CameraBuilder::setNear(float nearDistance) {
    this->near = nearDistance;
    return *this;
}

CameraBuilder &CameraBuilder::setFar(float farDistance) {
    this->far = farDistance;
    return *this;
}

Camera CameraBuilder::build() {
    return Camera{tf, fov, aspectRatio, aperture, focusDist, near, far};
}
