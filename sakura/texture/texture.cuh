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

    //    __host__ __device__ Texture() noexcept = default;
    __host__ __device__ Texture(const Texture &other) noexcept;
    __host__ __device__ Texture(Texture &&other) noexcept;
    __host__ __device__ Texture &operator=(const Texture &other) noexcept;
    __host__ __device__ Texture &operator=(Texture &&other) noexcept;
    __host__ __device__ ~Texture() noexcept = default;

    __host__ explicit Texture() noexcept;

    __host__ explicit Texture(const std::filesystem::path &path, bool isEnvMap = false, Eigen::Affine3f transform = Eigen::Affine3f::Identity()) noexcept(false);

    [[nodiscard]] __host__ static Texture RandomConstant() noexcept;

    [[nodiscard]] __host__ static Texture ZERO() noexcept;
    [[nodiscard]] __host__ static Texture DEFAULT() noexcept;

    [[nodiscard]] __host__ __device__ explicit Texture(Color color) noexcept;

    [[nodiscard]] __host__ __device__ float pdf(size_t idx) const noexcept;
    [[nodiscard]] __host__ __device__ float pdf(const EmitterQueryRecord &emitterQueryRecord) const noexcept;


    [[nodiscard]] __host__ __device__ Color eval(const Eigen::Vector2f &uv) const noexcept;
    [[nodiscard]] __host__ __device__ Color eval(const Ray &ray) const noexcept;

    [[nodiscard]] __host__ __device__ Color sample(EmitterQueryRecord &emitterQueryRecord, const Eigen::Vector3f &sample) const noexcept;


private:
    union {
        Color constant = {0.5f, 0.1f, 0.1f};
        struct {
            Vec3f *texture;
            float *cdf;
            int width, height;
            Eigen::Affine3f inverseTransform;
        } image;
    };
};