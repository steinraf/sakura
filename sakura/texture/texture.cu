//
// Created by steinraf on 26.02.25.
//

#include <utility>

#include "texture.cuh"
__host__ __device__ Color Texture::eval(const Eigen::Vector2f &uv) const noexcept {
    switch(type) {
        case TextureType::CONSTANT:
            return constant;
        default:
            assert(false);
            return {0.0f, 0.0f, 0.0f};
    }
}
__host__ __device__ Texture::Texture(Color color) noexcept
    : type(TextureType::CONSTANT), constant{std::move(color)} {
}
__host__ __device__ Texture::Texture(const Texture &other) noexcept {
    switch(other.type) {
        case TextureType::CONSTANT:
            constant = other.constant;
            break;
        default:
            assert(false);
            break;
    }
}
__host__ __device__ Texture::Texture(Texture &&other) noexcept {
    switch(other.type) {
        case TextureType::CONSTANT:
            constant = other.constant;
            break;
        default:
            assert(false);
            break;
    }
}
__host__ __device__ Texture &Texture::operator=(const Texture &other) noexcept {
    switch(other.type) {
        case TextureType::CONSTANT:
            constant = other.constant;
            break;
        default:
            assert(false);
            break;
    }
    return *this;
}
__host__ __device__ Texture &Texture::operator=(Texture &&other) noexcept {
    switch(other.type) {
        case TextureType::CONSTANT:
            constant = other.constant;
            break;
        default:
            assert(false);
            break;
    }
    return *this;
}
