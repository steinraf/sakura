//
// Created by steinraf on 21/08/22.
//

#include "../utility/warp.h"
#include "camera.h"

__device__ __host__ Camera::Camera(Vector3f origin, Vector3f lookAt, Vector3f _up, float vFOV,
                                   float aspectRatio, float aperture, float focusDist, float k1, float k2)
    : origin(origin), lensRadius(aperture * aspectRatio * sqrtf(2.f) / 2.0f /* scaled to fit mitsuba */), focusDist(focusDist), k1(k1), k2(k2) {

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

__device__ Ray3f Camera::getRay(float s, float t, const Vector2f &sample) const {

    const float distSq = (s - 0.5f)*(s - 0.5f) * (t - 0.5f)*(t - 0.5f);
    const float distortion = (k1 * distSq + k2*distSq*distSq);

    s += (s-0.5f)*distortion;
    t += (t-0.5f)*distortion;


    //Change aperture here by commenting out the correct aperture shape

    //Circular Aperture
    const Vector2f apertureSample = lensRadius * Warp::squareToUniformDisk(sample);

    //Square Aperture
    //        const Vector2f apertureSample = lensRadius * (Warp::squareToUniformSquare(sample) - Vector2f{0.5f});

    //Triangular Aperture
    //        const Vector3f triaSample =  Warp::squareToUniformTriangle(sample);
//        const Vector3f triaPoint = lensRadius * (   Vector3f{-0.5, -0.5, 0} * triaSample[0] +
//                                                    Vector3f{-0.5,  0.5, 0} * triaSample[1] +
//                                                    Vector3f{ 0.5,  0.5, 0} * triaSample[2] );
//        const Vector2f apertureSample{triaPoint[0], triaPoint[1]};

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