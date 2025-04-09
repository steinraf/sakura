//
// Created by steinraf on 10.04.25.
//

#include "camera.cuh"

CPU_GPU Camera::Camera() noexcept
    : Camera(Eigen::Isometry3f::Identity(), 90.f, 16.f / 9.f, 0.0f, 1.0f) {}

CPU_GPU Camera::Camera(Eigen::Isometry3f tf, float fov,
                       float aspectRatio, float aperture,
                       float focusDist, float near,
                       float far, ApertureType apertureType) noexcept
    : focusDist(focusDist),
      cameraTransform(std::move(tf)),
      sampleToCamera(),
      k(),
      near(near),
      far(far),
      lensRadius(),
      aspectRatio(aspectRatio),
      apertureType(apertureType) {


    updateLensRadius(aperture);
    setK(fov);
    generateSampleToCameraMatrix();
}

CPU_GPU Camera::Camera(Vector3f origin, Vector3f lookAt, Vector3f _up, float vFOV, float aspectRatio, float aperture, float focusDist, float k1, float k2, ApertureType apertureType)
    : Camera(Camera::lookAt(origin, lookAt, _up), vFOV, aspectRatio, aperture, focusDist, 0.1f, 10000.0f, apertureType) {

}
__device__ Ray3f Camera::getRay(float u, float v, const Vector2f &sample) const {
    Eigen::Vector4f nearSample = (sampleToCamera * Eigen::Vector4f{u, v, 0.0, 1.0});

    assert(nearSample[3] != 0.0f);

    Vec3f nearP = Vec3f{nearSample[0] / nearSample[3],
                        nearSample[1] / nearSample[3],
                        nearSample[2] / nearSample[3]};

    const auto apertureSample = [&]()-> Vector2f {
        switch(apertureType){
            case ApertureType::Circular:
                return lensRadius * Warp::squareToUniformDisk(sample);
            case ApertureType::Triangular:
                return [&]() -> Vector2f {
                    const Vector3f triaSample =  Warp::squareToUniformTriangle(sample);
                    const Vector3f triaPoint = lensRadius * (   Vector3f{-0.5, -0.5, 0} * triaSample[0] +
                                                             Vector3f{-0.5,  0.5, 0} * triaSample[1] +
                                                             Vector3f{ 0.5,  0.5, 0} * triaSample[2] );
                    return {triaPoint[0], triaPoint[1]};
                }();
            case ApertureType::Square:

                return lensRadius * (Warp::squareToUniformSquare(sample) - Vector2f{0.5f});
            default:
                assert(false && "Unknown aperture type");
                return Vector2f{0.f, 0.f};
        }
    }();

    Vec3f pLens =
            lensRadius * Vec3f{apertureSample[0], apertureSample[1], 0.0};

    float ft = focusDist / nearP[2];
    Vec3f pFocus = ft * nearP;
    Vec3f deeNorm = (pFocus - pLens).normalized();
    Eigen::Vector3f dNorm = {deeNorm[0], deeNorm[1], deeNorm[2]};
    Vec3f dir = (cameraTransform.linear() * dNorm).normalized();

    if(!isfinite(dir[0]) || !isfinite(dir[1]) || !isfinite(dir[2])) {
        assert(false);
    }

    return Ray3f{cameraTransform * static_cast<Eigen::Vector3f>(pLens),
                 dir, near, far};
}

CPU_GPU void Camera::translate(const Vec3f &x) {
    cameraTransform.translation() += static_cast<Eigen::Vector3f>(x);
}


CPU_GPU void Camera::translateRelative(const Vec3f &x) {
    cameraTransform.translation() += cameraTransform.linear() * static_cast<Eigen::Vector3f>(x);
}
CPU_GPU Eigen::Isometry3f Camera::lookAt(const Vec3f &center, const Vec3f &lookAt, const Vec3f &up) {
    Vec3f f = (lookAt - center).normalized();
    Vec3f r = up.cross(-f).normalized();
    Vec3f u = -f.cross(r);

    Eigen::Isometry3f tf = Eigen::Isometry3f::Identity();
    tf.linear().col(0) = static_cast<Eigen::Vector3f>(r);
    tf.linear().col(1) = static_cast<Eigen::Vector3f>(u);
    tf.linear().col(2) = static_cast<Eigen::Vector3f>(f);
    tf.translation() = static_cast<Eigen::Vector3f>(center);

    return tf;
}
CPU_GPU void Camera::generateSampleToCameraMatrix() {
    sampleToCamera.matrix() << 2 * k, 0.f, 0.f, -k,
            0.f, -2 * k / aspectRatio, 0.f, k / aspectRatio,
            0.f, 0.f, 0.f, 1.f,
            0.f, 0.f, (near - far) / (near * far), 1.f / near;
}
CPU_GPU void Camera::setK(float fov) {
    k = tanf(fov * M_PIf / 360.f);
}
CPU_GPU void Camera::updateFOV(float fov) {
    setK(fov);
    generateSampleToCameraMatrix();
}
CPU_GPU void Camera::updateLensRadius(float aperture) {
    lensRadius = sqrtf(2.f) / 2.f * aperture * aspectRatio;
}
CPU_GPU void Camera::setFocusPlane(const Vec3f &point) {
    focusDist = (static_cast<Eigen::Vector3f>(point) - cameraTransform.translation()).norm();
}

