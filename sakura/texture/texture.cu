//
// Created by steinraf on 26.02.25.
//

#include <thrust/device_ptr.h>
#include <thrust/transform_reduce.h>
#include <thrust/transform_scan.h>
#include <utility>


#include "texture.cuh"

#include "../emitter/emitter.cuh"
#include "../geometry/ray.cuh"

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#define TINYEXR_USE_STB_ZLIB 1
#define TINYEXR_IMPLEMENTATION
#include "tinyexr.h"

#define IMG_EPS 1e-4f


__host__ __device__ Color Texture::eval(const Eigen::Vector2f &uv) const noexcept {
    switch(type) {
        case TextureType::CONSTANT:
            return constant;
        case TextureType::IMAGE:
            return [&]() -> Color {
                const float x = std::clamp(uv.x() * (image.width - 1.0f), 0.0f, image.width - 1 - IMG_EPS),
                            y = std::clamp(image.height - 1 - uv.y() * (image.height - 1.0f), 0.0f, image.height - 1 - IMG_EPS);

                const int x1 = std::floor(x), x2 = x1 + 1;
                const int y1 = std::floor(y), y2 = y1 + 1;

#ifndef NDEBUG
                if(x1 < 0 || x2 >= image.width) printf("x out of bounds for uv (%f %f) in image texture eval (%u %u)\n", uv.x(), uv.y(), x1, x2);
                if(x1 < 0 || x2 >= image.width) assert(!"x out of bounds in image texture eval");
                if(y1 < 0 || y2 >= image.height) printf("y out of bounds for uv (%f %f) in image texture eval (%u %u)\n", uv.x(), uv.y(), y1, y2);
                if(y1 < 0 || y2 >= image.height) assert(!"y out of bounds in image texture eval");
#endif

                // Bilinear interpolation
                const Color c00 = image.texture[y1 * image.width + x1],
                            c01 = image.texture[y1 * image.width + x2],
                            c10 = image.texture[y2 * image.width + x1],
                            c11 = image.texture[y2 * image.width + x2];

                const float w00 = (x2 - x) * (y2 - y),
                            w01 = (x - x1) * (y2 - y),
                            w10 = (x2 - x) * (y - y1),
                            w11 = (x - x1) * (y - y1);


                return w00 * c00 + w01 * c01 +
                       w10 * c10 + w11 * c11;
            }();
        default:
            assert(false);
            return {0.0f, 0.0f, 0.0f};
    }
}
__host__ __device__ Texture::Texture(Color color) noexcept
    : type(TextureType::CONSTANT), constant{std::move(color)} {
}
__host__ __device__ Texture::Texture(const Texture &other) noexcept : type(other.type) {
    switch(other.type) {
        case TextureType::CONSTANT:
            constant = other.constant;
            break;
        case TextureType::IMAGE:
            image = other.image;
            break;
        default:
            assert(false);
            break;
    }
}
__host__ __device__ Texture::Texture(Texture &&other) noexcept : type(other.type) {
    switch(other.type) {
        case TextureType::CONSTANT:
            constant = other.constant;
            break;
        case TextureType::IMAGE:
            image = other.image;
            break;
        default:
            assert(false);
            break;
    }
}
__host__ __device__ Texture &Texture::operator=(const Texture &other) noexcept {
    type = other.type;
    switch(other.type) {
        case TextureType::CONSTANT:
            constant = other.constant;
            break;
        case TextureType::IMAGE:
            image = other.image;
            break;
        default:
            assert(false);
            break;
    }
    return *this;
}
__host__ __device__ Texture &Texture::operator=(Texture &&other) noexcept {
    type = other.type;
    switch(other.type) {
        case TextureType::CONSTANT:
            constant = other.constant;
            break;
        case TextureType::IMAGE:
            image = other.image;
            break;
        default:
            assert(false);
            break;
    }
    return *this;
}


struct ColorToRadiance {
    const Vec3f *const data;
    const int width, height;
    const bool isEnvMap;

    __host__ __device__ float operator()(const Vec3f &v) const noexcept {
        const auto y = static_cast<float>((&v - data) % width);
        if(isEnvMap) {
            return std::clamp(v.norm(), 0.0f, 100.0f) * std::sin(y / static_cast<float>(height));
        } else {
            return std::clamp(v.norm(), 0.0f, 100.0f);
        }
    }
};


struct ColorToCDF {
    const Vec3f *const data;
    const int width, height;
    const float sum;
    const bool isEnvMap;

    __host__ __device__ float operator()(const Vec3f &v) const noexcept {
        const auto y = static_cast<float>((&v - data) % width);
        if(isEnvMap) {
            return std::clamp(v.norm(), 0.0f, 1000.0f) * std::sin(y / static_cast<float>(height)) / sum;
        } else {
            return std::clamp(v.norm(), 0.0f, 1000.0f) / sum;
        }
    }
};

__host__ Texture::Texture(const std::filesystem::path &path, bool isEnvMap, Eigen::Affine3f transform) noexcept(false)
    : type(TextureType::IMAGE) {

    int dim = image.width = image.height = 0;

    const auto filedtype = path.extension().string();

    float *hostTexture = nullptr;

    auto *stbTexture = stbi_load(path.c_str(), &image.width, &image.height, &dim, 3);

    if(stbTexture) {
        hostTexture = new float[image.width * image.height * 3];
        for(int i = 0; i < image.width * image.height; i++) {
            hostTexture[i * 3 + 0] = stbTexture[i * 3 + 0] / 255.0f;
            hostTexture[i * 3 + 1] = stbTexture[i * 3 + 1] / 255.0f;
            hostTexture[i * 3 + 2] = stbTexture[i * 3 + 2] / 255.0f;
        }
        stbi_image_free(stbTexture);
    } else {
        const char *err = nullptr;
        int ret = LoadEXR(&hostTexture, &image.width, &image.height, path.c_str(), &err);
        if(ret != TINYEXR_SUCCESS) {
            throw std::runtime_error("Failed to load texture because of " + std::string(err));
        }
        //remove alpha channel
        for(int i = 0; i < image.width * image.height; i++) {
            hostTexture[i * 3 + 0] = hostTexture[i * 4 + 0];
            hostTexture[i * 3 + 1] = hostTexture[i * 4 + 1];
            hostTexture[i * 3 + 2] = hostTexture[i * 4 + 2];
        }

        dim = 3;
    }

    std::cout << "Image dims " << image.width << " " << image.height << std::endl;

    image.inverseTransform = transform.inverse();

    assert(dim == 3);

    checkCudaErrors(cudaMallocManaged(&image.texture, image.width * image.height * sizeof(Vec3f)));
    checkCudaErrors(cudaMallocManaged(&image.cdf, image.width * image.height * sizeof(float)));

    checkCudaErrors(cudaMemcpy(image.texture, hostTexture, image.width * image.height * sizeof(Vec3f), cudaMemcpyHostToDevice));


    ColorToRadiance colorToRadiance{image.texture, image.width, image.height, isEnvMap};
    thrust::device_ptr<Vec3f> devPtr(image.texture);
    float cdfSum = thrust::transform_reduce(devPtr, devPtr + image.width * image.height, colorToRadiance, 0.0f, thrust::plus<float>());

    ColorToCDF colorToCdf{image.texture, image.width, image.height, cdfSum, isEnvMap};
    thrust::transform_inclusive_scan(devPtr, devPtr + image.width * image.height, image.cdf, colorToCdf, thrust::plus<float>());
}
__host__ __device__ float Texture::pdf(size_t idx) const noexcept {
    switch(type) {
        case TextureType::CONSTANT:
            return 1.0f;
        case TextureType::IMAGE:
            if(idx == image.width * image.height - 1)
                return std::max(IMG_EPS, 1.0f - image.cdf[idx]);
            return std::max(IMG_EPS, image.cdf[idx + 1] - image.cdf[idx]);
        default:
            assert(false);
            return 0.0f;
    }
}
__host__ Texture::Texture() noexcept {
    auto t = DEFAULT();
    type = t.type;
    constant = t.constant;
}
__host__ Texture Texture::RandomConstant() noexcept {
    return Texture{Eigen::Vector3f::Random().normalized().cwiseAbs()};
}

__host__ Texture Texture::ZERO() noexcept {
    Texture t{Color::Zero()};
    return t;
}

__host__ Texture Texture::DEFAULT() noexcept {
    Texture t{{255.f / 255, 183.f / 255, 197.f / 255}};
    return t;
}
__host__ __device__ float Texture::pdf(const EmitterQueryRecord &emitterQueryRecord) const noexcept {
    const float pdf = this->pdf(emitterQueryRecord.idx);
    const float k = std::max(IMG_EPS, 2 * std::sin(M_PIf * emitterQueryRecord.uv[1]));
    return pdf * M_1_PIf * M_1_PIf / k;
}
__host__ __device__ Color Texture::eval(const Ray &_ray) const noexcept {
    Ray ray = _ray;
    ray.transform(image.inverseTransform);
    const Vec3f dir = ray.dir.normalized();

    const float u = std::atan2(dir[0], -dir[2]) * 0.5f * M_1_PIf;
    const float v = std::clamp(std::acos(-dir[1]) * M_1_PIf, -1.f, 1.f);

    if(!isfinite(u) || !isfinite(v)) {
        //        assert(false);
        return Color::Zero();
    }

    return eval(Eigen::Vector2f{(u < 0) ? (u + 1) : u, v});
}
__host__ __device__ Color Texture::sample(EmitterQueryRecord &emitterQueryRecord, const Eigen::Vector3f &sample) const noexcept {
    switch(type) {
        case TextureType::CONSTANT:
            return constant;
        case TextureType::IMAGE:
            return [&]() -> Color {
                auto idx = sample::sampleCDF(sample[2], image.cdf, image.width * image.height);


                emitterQueryRecord.idx = idx;

                const float u = float(idx % image.width) / image.width,
                            v = float(idx / image.width) / image.height;

                assert(u >= 0 && u <= 1);
                assert(v >= 0 && v <= 1);


                //TODO handle image.inverseTransform rotation
                const Vec3f dir = sample::squareToUniformSphere({u, v});


                emitterQueryRecord.point = emitterQueryRecord.point + dir * EMITTER_DIST();// Emitter at "infinite" distance
                emitterQueryRecord.wIn = -dir;
                emitterQueryRecord.shadowRay = Ray{emitterQueryRecord.point, dir, RAY_EPSILON, 0.5f * EMITTER_DIST() - RAY_EPSILON};
                emitterQueryRecord.uv = {u, v};

                const float pdf = this->pdf(emitterQueryRecord);
                emitterQueryRecord.pdf = pdf;
                assert(pdf > 0.0f);
                return eval(emitterQueryRecord.uv) / pdf;
            }();
        default:
            assert(false);
            return {0.0f, 0.0f, 0.0f};
    }
}
__host__ __device__ Color Texture::getConstant() const noexcept {
    assert(type == TextureType::CONSTANT);
    return constant;
}
__host__ Texture Texture::ONES() noexcept {
    Texture t{Color::Ones()};
    return t;
}
