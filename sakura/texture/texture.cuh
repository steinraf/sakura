//
// Created by steinraf on 26.02.25.
//

#pragma once

#include "../common.cuh"


enum class TextureType {
    CONSTANT,
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

    __host__ explicit Texture() noexcept {
        auto t = DEFAULT();
        type = t.type;
        constant = t.constant;
    }

    [[nodiscard]] __host__ static Texture RandomConstant() noexcept {
        return Texture{Eigen::Vector3f::Random().normalized().cwiseAbs()};
    }

    [[nodiscard]] __host__ static Texture DEFAULT() noexcept {
        Texture t{{255.f / 255, 183.f / 255, 197.f / 255}};
        return t;
    }

    [[nodiscard]] __host__ __device__ explicit Texture(Color color) noexcept;

    [[nodiscard]] __host__ __device__ Color eval(const Eigen::Vector2f &uv) const noexcept;

private:
    union {
        Color constant = {0.5f, 0.1f, 0.1f};
    };
};