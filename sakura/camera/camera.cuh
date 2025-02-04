//
// Created by steinraf on 04.02.25.
//

#pragma once

#include <Eigen/Dense>

class Camera{
public:
    Camera();
    explicit Camera(const Eigen::Transform<float, 3, Eigen::Affine>& tf) : tf(tf) {}
    Eigen::Transform<float, 3, Eigen::Affine> tf;
};