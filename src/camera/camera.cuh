//
// Created by steinraf on 21/08/22.
//

#pragma once

#include "../common.h"

#include "../../src_old/utility/ray.h"
#include "../../src_old/utility/warp.h"


enum class ApertureType{
    Circular,
    Triangular,
    Square,
};

class Camera {
public:
    CPU_GPU_CONSTEXPR Camera(Vector3f origin, Vector3f lookAt, Vector3f _up, float vFOV, float aspectRatio,
                             float aperture, float focusDist, float k1, float k2, ApertureType apertureType);

    [[nodiscard]] CPU_GPU_CONSTEXPR Ray3f getRay(float s, float t, const Vector2f &sample) const;

    //relative velocity with right/up/front
    CPU_GPU_CONSTEXPR void addVelocityRelative(const Vector3f &v, float t) noexcept;

    CPU_GPU_CONSTEXPR  Vector3f getPosition() const noexcept;
private:
    Vector3f origin;
    Vector3f right, up, front;

    Matrix4f sampleToCamera;
    Matrix4f cameraToWorld;

    const float lensRadius;
    const float focusDist;
    const float k1, k2;
    const float near = 0.0001;
    const float far = 100000.f;


    ApertureType apertureType;
};

//Implementations

CPU_GPU_CONSTEXPR Camera::Camera(Vector3f origin,
                                             Vector3f lookAt,
                                             Vector3f _up,
                                             float vFOV,
                                             float aspectRatio,
                                             float aperture,
                                             float focusDist,
                                             float k1,
                                             float k2,
                                             ApertureType apertureType)
        : origin(origin),
          lensRadius(aperture * aspectRatio * std::sqrt(2.f) / 2.0f /* scaled to fit mitsuba */),
          focusDist(focusDist),
          k1(k1), k2(k2), apertureType(apertureType) {

    const float k = tan(vFOV * M_PIf / 360.f);


    sampleToCamera = Matrix4f{
            2*k   , 0.f           , 0.f                      , -k,
            0.f , -2*k/aspectRatio, 0.f                      , k/aspectRatio,
            0.f , 0.f           , 0.f                      , 1.f,
            0.f , 0.f           , (near-far)/(near*far)   , 1.f/near
    };


    constexpr int noriConvert = 1;// -1 for nori, 1 for correct handedness


    front = (lookAt - origin).normalized();
    right = noriConvert * (_up.cross(-front)).normalized();
    up = front.cross(noriConvert * -right).normalized();

    cameraToWorld = Matrix4f{
            right[0], up[0], front[0], origin[0],
            right[1], up[1], front[1], origin[1],
            right[2], up[2], front[2], origin[2],
            0.f, 0.f, 0.f, 1.f,
    };

}

CPU_GPU_CONSTEXPR Ray3f Camera::getRay(float s, float t, const Vector2f &sample) const{

    const float distSq = (s - 0.5f)*(s - 0.5f) * (t - 0.5f)*(t - 0.5f);
    const float distortion = (k1 * distSq + k2*distSq*distSq);

    s += (s-0.5f)*distortion;
    t += (t-0.5f)*distortion;

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

    const Vector3f nearP = (Vector3f{s, t, 0.f}.applyTransform(sampleToCamera)).normalized();
    Vector3f pLens(apertureSample[0], apertureSample[1], 0.f);
    float ft = focusDist / nearP[2];
    Vector3f pFocus = ft * nearP;
    Vector3f d_norm = (pFocus - pLens).normalized();

    return {
            pLens .applyTransform(cameraToWorld),
            (d_norm.applyTransform(cameraToWorld) - origin).normalized(),
            near,
            far
    };
}

CPU_GPU_CONSTEXPR void Camera::addVelocityRelative(const Vector3f &v, float t) noexcept {
    cameraToWorld.addPosition((v[0] * right + v[1] * up + v[2] * front)* t);
}

CPU_GPU_CONSTEXPR Vector3f Camera::getPosition() const noexcept{
    return cameraToWorld.getPosition();
}