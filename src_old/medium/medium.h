//
// Created by steinraf on 20/12/22. Ported from code written by joluther.
//

#pragma once


#include "../utility/vector.cuh"
#include "isotropicPhaseFunction.h"
class GalaxyMedium {
private:

    __device__ constexpr float galaxy(const Vector3f &p) const noexcept{
        if(isHomogeneous) return 1.f;

        Vector3f rel = p - m_center;

        float x = rel.dot(m_tangentX);
        float y = rel.dot(m_tangentY);
        float z = rel.dot(m_normal);

        float r = sqrtf(x*x+y*y) / m_radius;

        float spiralDensity = (powf((0.5f+0.5f*cos(m_armCount * atan2f(x,y) + r*m_twist)), m_armThinning*r*r)) * exp(-2*r*r/m_armLength/m_armLength);
        return exp(- m_armCount * z*z/spiralDensity / m_armDepth);// + 0.001 + exp(-2*rel.squaredNorm()) * 0.1;
    }

public:

    float m_sigmaA = 0.1f;
    float m_sigmaS = 0.1f;
    float m_radius = 3.0f;

    int m_armCount = 2;
    float m_armThinning = 10.f;
    float m_armDepth = 2.f;
    float m_armLength = 2.f;

    Vector3f m_center{0.1f};
    Vector3f m_normal{0.f, 0.f, 1.f};
    float m_twist = 20.f;

    Vector3f m_tangentX{1.f};
    Vector3f m_tangentY;

    bool isActive = false;
    bool isHomogeneous = false;

    IsotropicPhaseFunction phaseFunction{};


    [[nodiscard]] __device__ const IsotropicPhaseFunction& getPhaseFunction() const{
        return phaseFunction;
    }

    constexpr __device__ __host__ GalaxyMedium() noexcept
        :m_sigmaA(0.f), m_sigmaS(0.f) {
        m_tangentX = (m_tangentX - m_tangentX.dot(m_normal)*m_normal).normalized();
        m_tangentY = m_tangentX.cross(m_normal);
    }

    [[nodiscard]] __device__ __host__ float getExtinction(const Vector3f &p) const noexcept{
        return (m_sigmaA + m_sigmaS) * galaxy(p);
    }

    [[nodiscard]] __device__ __host__ float getScattering(const Vector3f &p) const noexcept{
        return m_sigmaS * galaxy(p);
    }

    [[nodiscard]] __device__ __host__ float getMaxExtinction() const noexcept {
        return m_sigmaA + m_sigmaS;
    }

    [[nodiscard]] __device__ __host__ Color3f getAlbedo(const Vector3f &p) const noexcept{
        return Color3f{1.f}; //Color3f(Color3f{1.0, 0.0, 0.0} + 0.5*Color3f{galaxy(p)}).clamp(0.f,1.f);
    }
};
