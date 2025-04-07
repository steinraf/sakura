//
// Created by steinraf on 21/08/22.
//

#pragma once

#include "../../src/common.h"

#include "utility/frame.h"
#include "utility/ray.h"
#include "utility/vector.cuh"


struct Intersection {
    Vector3f p;
    Frame shFrame;

    Vector2f uv;

    BLAS const *mesh = nullptr;

    float t = 0.f;

    constexpr Intersection() = default;
};
