//
// Created by steinraf on 26.02.25.
//

#pragma once

#include "../common.cuh"
#include <filesystem>


enum class TextureType {
    CONSTANT,
    IMAGE,
};

class Texture {
public:
    TextureType type = TextureType::CONSTANT;

    [[nodiscard]] CPU_GPU static float EMITTER_DIST() noexcept { return 1000000.0f; }

    //    CPU_GPU Texture() noexcept = default;
    CPU_GPU Texture(const Texture &other) noexcept;
    CPU_GPU Texture(Texture &&other) noexcept;
    CPU_GPU Texture &operator=(const Texture &other) noexcept;
    CPU_GPU Texture &operator=(Texture &&other) noexcept;
    CPU_GPU ~Texture() noexcept = default;

    __host__ explicit Texture() noexcept;

    __host__ explicit Texture(const std::filesystem::path &path, bool isEnvMap = false, Eigen::Affine3f transform = Eigen::Affine3f::Identity()) noexcept(false);

    [[nodiscard]] __host__ static Texture RandomConstant() noexcept;

    [[nodiscard]] __host__ static Texture ZERO() noexcept;
    [[nodiscard]] __host__ static Texture DEFAULT() noexcept;
    [[nodiscard]] __host__ static Texture ONES() noexcept;

    [[nodiscard]] CPU_GPU explicit Texture(Color color) noexcept;

    [[nodiscard]] CPU_GPU float pdf(size_t idx) const noexcept;
    [[nodiscard]] CPU_GPU float pdf(const EmitterQueryRecord &emitterQueryRecord) const noexcept;


    [[nodiscard]] CPU_GPU Color eval(const Vec2f &uv) const noexcept;
    [[nodiscard]] CPU_GPU Color eval(const Ray &ray) const noexcept;

    [[nodiscard]] CPU_GPU Color sample(EmitterQueryRecord &emitterQueryRecord, const Vec3f &sample) const noexcept;

    [[nodiscard]] CPU_GPU Color getConstant() const noexcept;

    union {
        Color constant = {0.5f, 0.1f, 0.1f};
        struct {
            Vec3f *texture;
            float *cdf;
            int width, height;
            Eigen::Affine3f inverseTransform;
        } image;
    };

private:
};