//
// Created by steinraf on 04.02.25.
//

#pragma once

#include <Eigen/Dense>

struct Intersection{
    Eigen::Vector3f point;

    Eigen::Vector2f uv;
    float t;
};