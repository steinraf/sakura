#pragma once

#define EPSILON 0.0001f



#define CPU_GPU __host__ __device__
#define CPU_GPU_INLINE __host__ __device__ inline
#define CPU_GPU_CONSTEXPR __host__ __device__ inline constexpr
#define CPU_ONLY __host__
#define GPU_ONLY __device__


#include <charconv>
#include <cmath>
#include <cstdlib>
#include <cuda/std/cassert>
#include <iomanip>
#include <iostream>
#include <sstream>

#include <cuda_runtime.h>

#include <eigen3/Eigen/Dense>

namespace Warp{
    [[nodiscard]] CPU_GPU_CONSTEXPR float gammaCorrect(float value) noexcept {
        if(value <= 0.0031308f) return std::clamp(12.92f * value, 0.f, 1.f);
        return std::clamp(1.055f * std::pow(value, 1.f / 2.4f) - 0.055f, 0.f, 1.f);
    }
}

class Vector3f {
public:
    CPU_GPU_CONSTEXPR Vector3f() noexcept : data{0.0f, 0.0f, 0.0f} {}

    CPU_GPU_CONSTEXPR Vector3f(float x, float y, float z) noexcept : data{x, y, z} {}

    CPU_GPU_CONSTEXPR explicit Vector3f(float v) noexcept : data{v, v, v} {}

    //support constructor from eigen
    CPU_GPU Vector3f(const Eigen::Vector3f &v) noexcept
        : data{v[0], v[1], v[2]} {
    }

    __host__ constexpr explicit Vector3f(const std::string_view &str) : data{0.f, 0.f, 0.f} {

        auto checkConversion = [](std::from_chars_result res, float result) {
            auto [ptr, ec] = res;
            if(ec == std::errc::invalid_argument || ec == std::errc::result_out_of_range) {
                throw std::runtime_error("Invalid argument in vector construction from string.");
            }
        };

        std::string_view currentString(str);

        for(float & i : data){
            checkConversion(std::from_chars(currentString.begin(), currentString.end(), i), i);
            currentString.remove_prefix(currentString.find(',') + 1);
            currentString.remove_prefix(currentString.find_first_not_of(" \r\n\t\v\f"));
        }
    }



    //support cast to eigen
    [[nodiscard]] CPU_GPU operator Eigen::Vector3f() const noexcept {
        return Eigen::Vector3f{data[0], data[1], data[2]};
    }



    [[nodiscard]] __host__ __device__ constexpr inline float operator[](int i) const noexcept { return data[i]; }

    [[nodiscard]] __host__ __device__ constexpr inline float &operator[](int i) noexcept { return data[i]; };

    [[nodiscard]] __host__ __device__ constexpr inline Vector3f operator-() const noexcept;

    [[nodiscard]] __host__ __device__ constexpr inline Vector3f operator+(const Vector3f &v2) const noexcept;

    [[nodiscard]] __host__ __device__ constexpr inline Vector3f operator-(const Vector3f &v2) const noexcept;

    [[nodiscard]] __host__ __device__ constexpr inline Vector3f operator*(const Vector3f &v2) const noexcept;

    [[nodiscard]] __host__ __device__ constexpr inline Vector3f operator/(const Vector3f &v2) const;

    [[nodiscard]] __host__ __device__ constexpr inline Vector3f operator*(float t) const noexcept;

    [[nodiscard]] __host__ __device__ constexpr inline Vector3f operator/(float t) const;

    __host__ __device__ constexpr inline bool operator==(const Vector3f &v2) const noexcept;

    __host__ __device__ constexpr inline bool operator!=(const Vector3f &v2) const noexcept;

    __host__ __device__ constexpr inline Vector3f &operator+=(const Vector3f &v2) noexcept;

    __host__ __device__ constexpr inline Vector3f &operator-=(const Vector3f &v2) noexcept;

    __host__ __device__ constexpr inline Vector3f &operator*=(const Vector3f &v2) noexcept;

    __host__ __device__ constexpr inline Vector3f &operator/=(const Vector3f &v2);

    __host__ __device__ constexpr inline Vector3f &operator*=(float t) noexcept;

    __host__ __device__ constexpr inline Vector3f &operator/=(float t);

    [[nodiscard]] __host__ __device__ constexpr inline float norm() const noexcept;

    [[nodiscard]] __host__ __device__ constexpr inline float squaredNorm() const noexcept;

    [[nodiscard]] __host__ __device__ constexpr inline Vector3f normalized() const;

    [[nodiscard]] __host__ __device__ constexpr inline float dot(const Vector3f &v2) const noexcept;

    [[nodiscard]] __host__ __device__ constexpr inline Vector3f cross(const Vector3f &v2) const;

    [[nodiscard]] __host__ __device__ constexpr inline int asColor(size_t i) const noexcept;

    [[nodiscard]] __host__ __device__ constexpr inline Vector3f
    applyTransform(const struct Matrix4f &transform, bool isTranslationInvariant = false) const noexcept;


    __host__ __device__ constexpr inline Vector3f &clamp(float minimum, float maximum) noexcept;

    [[nodiscard]] __host__ __device__ constexpr inline Vector3f absValues() const noexcept;

    [[nodiscard]] __host__ __device__ constexpr inline bool isZero() const noexcept;

    [[nodiscard]] __host__ __device__ constexpr inline float maxCoeff() const noexcept;
    [[nodiscard]] __host__ __device__ constexpr inline float minCoeff() const noexcept;


    [[nodiscard]] __host__ __device__ constexpr inline bool isValid() const noexcept{
        return std::isfinite(data[0]) && std::isfinite(data[1]) && std::isfinite(data[2]);
    }

    [[nodiscard]] __host__ __device__ constexpr inline Vector3f gammaCorrected() const noexcept;

    [[nodiscard]] __host__ __device__ constexpr static inline Vector3f Zero() noexcept {
        return {0.f, 0.f, 0.f};
    }


//    __device__ static inline void atomicCudaAdd(Vector3f *address, const Vector3f &vec) noexcept {
//        Vector3f &v = *address;
//        atomicAdd(&(v[0]), vec[0]);
//        atomicAdd(&(v[1]), vec[1]);
//        atomicAdd(&(v[2]), vec[2]);
//    }




    friend inline std::ostream &operator<<(std::ostream &os, const Vector3f &t);


private:
    float data[3];
};

using Color3f = Vector3f;

class Matrix3f {

public:
    __host__ __device__ constexpr Matrix3f(const Vector3f &row1, const Vector3f &row2, const Vector3f &row3) noexcept
        : row1(row1), row2(row2), row3(row3) {
    }


    [[nodiscard]] __host__ __device__ constexpr Vector3f operator*(const Vector3f &vec) const noexcept {
        return {
                row1.dot(vec),
                row2.dot(vec),
                row3.dot(vec)};
    }

    [[nodiscard]] __host__ __device__ constexpr Matrix3f operator*(const Matrix3f &mat) const noexcept {
        const Vector3f c1 = {mat.row1[0], mat.row2[0], mat.row3[0]},
                       c2 = {mat.row1[1], mat.row2[1], mat.row3[1]},
                       c3 = {mat.row1[2], mat.row2[2], mat.row3[2]};
        return {
                {row1.dot(c1), row1.dot(c2), row1.dot(c3)},
                {row2.dot(c1), row2.dot(c2), row2.dot(c3)},
                {row3.dot(c1), row3.dot(c2), row3.dot(c3)}};
    }

    __host__ __device__ constexpr Matrix3f(const Vector3f axis, float angle) noexcept {
        float theta = angle * M_PIf / 180.f;

        auto u = axis.normalized();
        const float cosT = cos(theta), sinT = sin(theta);
        assert(theta >= -2 * M_PIf && theta <= 2 * M_PIf && "rotation should be in radians from [-2PI,2PI]");
        *this = Matrix3f{
                {cosT + u[0] * u[0] * (1 - cosT), u[0] * u[1] * (1 - cosT) - u[2] * sinT, u[0] * u[2] * (1 - cosT) + u[1] * sinT},
                {u[0] * u[1] * (1 - cosT) + u[2] * sinT, cosT + u[1] * u[1] * (1 - cosT), u[1] * u[2] * (1 - cosT) - u[0] * sinT},
                {u[0] * u[2] * (1 - cosT) - u[1] * sinT, u[1] * u[2] * (1 - cosT) + u[0] * sinT, cosT + u[2] * u[2] * (1 - cosT)}};
    }

    [[nodiscard]] __host__ __device__ static constexpr inline Matrix3f fromDiag(const Vector3f &vec) noexcept {
        return {
                {vec[0], 0, 0},
                {0, vec[1], 0},
                {0, 0, vec[2]}};
    }

    [[nodiscard]] __host__ __device__ static constexpr inline Matrix3f makeIdentity() noexcept {
        return fromDiag(Vector3f{1.f});
    }

    [[nodiscard]] __host__ __device__ constexpr Vector3f operator[](size_t idx) const noexcept{
        switch(idx){
            case 0:
                return row1;
            case 1:
                return row2;
            case 2:
                return row3;
            default:
                assert(false && "Index out of bounds.");
                return row3;
        }
    }

private:
    Vector3f row1;
    Vector3f row2;
    Vector3f row3;
};

//TODO maybe change to quaternion representation
struct Matrix4f{
    __host__ __device__ constexpr Matrix4f() noexcept
        :mat{   {1.f, 0.f, 0.f, 0.f},
                {0.f, 1.f, 0.f, 0.f},
                {0.f, 0.f, 1.f, 0.f},
                {0.f, 0.f, 0.f, 1.f}
          }{
    }
    __host__ __device__ constexpr explicit Matrix4f(const Matrix3f &r) noexcept
        :mat{   {r[0][0], r[0][1], r[0][2], 0.f},
                {r[1][0], r[1][1], r[1][2], 0.f},
                {r[2][0], r[2][1], r[2][2], 0.f},
                {0.f, 0.f, 0.f, 1.f}
          }{
    }
    __host__ __device__ constexpr explicit Matrix4f(const Vector3f &t) noexcept
        :mat{   {1.f, 0.f, 0.f, t[0]},
                {0.f, 1.f, 0.f, t[1]},
                {0.f, 0.f, 1.f, t[2]},
                {0.f, 0.f, 0.f, 1.f }
          }{
    }

    __host__ __device__ constexpr Matrix4f(const Matrix3f &r, const Vector3f &t) noexcept
        :mat{   {r[0][0], r[0][1], r[0][2], t[0]},
                {r[1][0], r[1][1], r[1][2], t[1]},
                {r[2][0], r[2][1], r[2][2], t[2]},
                {0.f, 0.f, 0.f, 1.f}
          }{
    }

    __host__ __device__ constexpr explicit Matrix4f(float a[4][4]) noexcept
        :mat{   {a[0][0], a[0][1], a[0][2], a[0][3]},
                {a[1][0], a[1][1], a[1][2], a[1][3]},
                {a[2][0], a[2][1], a[2][2], a[2][3]},
                {a[3][0], a[3][1], a[3][2], a[3][3]}
          }{

    }

    __host__ __device__ constexpr Matrix4f(float m00, float m01, float m02, float m03,
                                           float m10, float m11, float m12, float m13,
                                           float m20, float m21, float m22, float m23,
                                           float m30, float m31, float m32, float m33) noexcept
            :mat{ {m00, m01, m02, m03},
                  {m10, m11, m12, m13},
                  {m20, m21, m22, m23},
                  {m30, m31, m32, m33}}{

    }

    __host__ __device__ constexpr Matrix4f operator*(const Matrix4f &o) const noexcept{
        float matrix[4][4] {};
        for (int i = 0; i < 4; ++i) {
            for(int j = 0; j < 4; ++j) {
                float num = 0;
                for(int k = 0; k < 4; ++k) {
                    num += mat[i][k] * o.mat[k][j];
                }
                matrix[i][j] = num;
            }
        }
        return Matrix4f{matrix};
    }

    __host__ __device__ constexpr Vector3f getPosition() const noexcept {
        return Vector3f{
            mat[0][3],
            mat[1][3],
            mat[2][3]
        };
    }

    __host__ __device__ constexpr void addPosition(const Vector3f &pos) noexcept {
        mat[0][3] += pos[0];
        mat[1][3] += pos[1];
        mat[2][3] += pos[2];
    }

//private:
    float mat[4][4];
};

__host__ __device__ constexpr inline Vector3f operator*(float t, const Vector3f &v) noexcept;


inline std::ostream &operator<<(std::ostream &os, const Vector3f &t) {
    os << t.asColor(0) << ' '
       << t.asColor(1) << ' '
       << t.asColor(2) << '\n';
    return os;
}


__host__ __device__ constexpr inline int Vector3f::asColor(size_t i) const noexcept {
    return int(255.99 * data[i]);
}

__host__ __device__ constexpr inline float Vector3f::norm() const noexcept {
    return std::sqrt(squaredNorm());
}

__host__ __device__ constexpr inline float Vector3f::squaredNorm() const noexcept {
    return data[0] * data[0] + data[1] * data[1] + data[2] * data[2];
}


__host__ __device__ constexpr inline Vector3f Vector3f::normalized() const {
    float n = norm();
#ifndef NDEBUG
    if(n == 0) {
        printf("ERROR, Zero Vector is being normalized\n ");
        return {1.f, 0.f, 0.f};
    }
#endif
    float k = 1.f / n;
    return {
            data[0] * k,
            data[1] * k,
            data[2] * k};
}

__host__ __device__ constexpr inline Vector3f Vector3f::operator-() const noexcept {
    return {-data[0], -data[1], -data[2]};
}


__host__ __device__ constexpr inline Vector3f Vector3f::operator+(const Vector3f &v2) const noexcept {
    return {data[0] + v2.data[0],
            data[1] + v2.data[1],
            data[2] + v2.data[2]};
}

__host__ __device__ constexpr inline Vector3f Vector3f::operator-(const Vector3f &v2) const noexcept {
    return {data[0] - v2.data[0],
            data[1] - v2.data[1],
            data[2] - v2.data[2]};
}

__host__ __device__ constexpr inline Vector3f Vector3f::operator*(const Vector3f &v2) const noexcept {
    return {data[0] * v2.data[0],
            data[1] * v2.data[1],
            data[2] * v2.data[2]};
}

__host__ __device__ constexpr inline Vector3f Vector3f::operator/(const Vector3f &v2) const {
    assert(v2.data[0] != 0);
    assert(v2.data[1] != 0);
    assert(v2.data[2] != 0);
    return {data[0] / v2.data[0],
            data[1] / v2.data[1],
            data[2] / v2.data[2]};
}

__host__ __device__ constexpr inline Vector3f operator*(float t, const Vector3f &v) noexcept {
    return v * t;
}

__host__ __device__ constexpr inline Vector3f Vector3f::operator*(float t) const noexcept {
    return {t * data[0],
            t * data[1],
            t * data[2]};
}


__host__ __device__ constexpr inline Vector3f Vector3f::operator/(float t) const {
    assert(t != 0);
    return {data[0] / t,
            data[1] / t,
            data[2] / t};
}


__host__ __device__ constexpr inline float Vector3f::dot(const Vector3f &v2) const noexcept {
    return data[0] * v2.data[0] +
           data[1] * v2.data[1] +
           data[2] * v2.data[2];
}

__host__ __device__ constexpr inline Vector3f Vector3f::cross(const Vector3f &v2) const {
    return {data[1] * v2.data[2] - data[2] * v2.data[1],
            data[2] * v2.data[0] - data[0] * v2.data[2],
            data[0] * v2.data[1] - data[1] * v2.data[0]};
}


__host__ __device__ constexpr inline Vector3f &Vector3f::operator+=(const Vector3f &v) noexcept {
    data[0] += v.data[0];
    data[1] += v.data[1];
    data[2] += v.data[2];
    return *this;
}

__host__ __device__ constexpr inline Vector3f &Vector3f::operator*=(const Vector3f &v) noexcept {
    data[0] *= v.data[0];
    data[1] *= v.data[1];
    data[2] *= v.data[2];
    return *this;
}

__host__ __device__ constexpr inline Vector3f &Vector3f::operator/=(const Vector3f &v) {
    assert(v.data[0] != 0);
    assert(v.data[1] != 0);
    assert(v.data[2] != 0);

    data[0] /= v.data[0];
    data[1] /= v.data[1];
    data[2] /= v.data[2];
    return *this;
}

__host__ __device__ constexpr inline Vector3f &Vector3f::operator-=(const Vector3f &v) noexcept {
    data[0] -= v.data[0];
    data[1] -= v.data[1];
    data[2] -= v.data[2];
    return *this;
}

__host__ __device__ constexpr inline Vector3f &Vector3f::operator*=(float t) noexcept {
    data[0] *= t;
    data[1] *= t;
    data[2] *= t;
    return *this;
}

__host__ __device__ constexpr inline Vector3f &Vector3f::operator/=(float t) {
    assert(t != 0);
    float k = 1.f / t;

    data[0] *= k;
    data[1] *= k;
    data[2] *= k;
    return *this;
}

__host__ __device__ constexpr inline Vector3f unit_vector(Vector3f v) {
    auto n = v.norm();
    assert(n != 0);
    return v / n;
}

__host__ __device__ constexpr inline Vector3f &Vector3f::clamp(float minimum, float maximum) noexcept {
    data[0] = std::max(minimum, std::min(maximum, data[0]));
    data[1] = std::max(minimum, std::min(maximum, data[1]));
    data[2] = std::max(minimum, std::min(maximum, data[2]));
    return *this;
}

__host__ __device__ constexpr inline Vector3f Vector3f::absValues() const noexcept {
    return {
            std::abs(data[0]),
            std::abs(data[1]),
            std::abs(data[2])
    };
}

__host__ __device__ constexpr inline bool Vector3f::isZero() const noexcept {
    return (data[0] == 0.f) &&
           (data[1] == 0.f) &&
           (data[2] == 0.f);
}

__host__ __device__ constexpr inline bool Vector3f::operator==(const Vector3f &v2) const noexcept {
    return data[0] == v2.data[0] && data[1] == v2.data[1] && data[2] == v2.data[2];
}

__host__ __device__ constexpr inline bool Vector3f::operator!=(const Vector3f &v2) const noexcept {
    return !(*this == v2);
}

__host__ __device__ constexpr Vector3f
Vector3f::applyTransform(const Matrix4f &transform, bool isTranslationInvariant) const noexcept {
    float v[4] {};

    for(int i = 0; i < 4; ++i){
        float sum = 0.f;
        for(int j = 0; j < 3; ++j){
            sum += transform.mat[i][j] * data[j];
        }
        if(!isTranslationInvariant)
            sum += transform.mat[i][3];
        v[i] = sum;
    }

    float divisor = 1.f;
    if(!isTranslationInvariant && v[3] != 0.f)
        divisor = 1.f/v[3];

    return Vector3f{v[0], v[1], v[2]}*divisor;
}

__host__ __device__ constexpr float Vector3f::maxCoeff() const noexcept {
    return std::max(data[0], std::max(data[1], data[2]));
}

__host__ __device__ constexpr float Vector3f::minCoeff() const noexcept {
    return std::min(data[0], std::min(data[1], data[2]));
}

[[nodiscard]] __host__ __device__ constexpr inline Vector3f Vector3f::gammaCorrected() const noexcept{
    return Vector3f{
            Warp::gammaCorrect(data[0]),
            Warp::gammaCorrect(data[1]),
            Warp::gammaCorrect(data[2]),
    };
}


class Vector2f {
public:
    __host__ __device__ constexpr Vector2f() noexcept : data{0.0f, 0.0f} {}

    __host__ __device__ constexpr Vector2f(float x, float y) noexcept : data{x, y} {}

    __host__ __device__ constexpr explicit Vector2f(float v) noexcept : data{v, v} {}

    [[nodiscard]] __host__ __device__ constexpr inline float operator[](int i) const noexcept { return data[i]; }

    [[nodiscard]] __host__ __device__ constexpr inline float &operator[](int i) noexcept { return data[i]; };

    [[nodiscard]] __host__ __device__ constexpr inline Vector2f operator-() const noexcept;

    [[nodiscard]] __host__ __device__ constexpr inline Vector2f operator+(const Vector2f &v2) const noexcept;

    [[nodiscard]] __host__ __device__ constexpr inline Vector2f operator-(const Vector2f &v2) const noexcept;

    [[nodiscard]] __host__ __device__ constexpr inline Vector2f operator*(const Vector2f &v2) const noexcept;

    [[nodiscard]] __host__ __device__ constexpr inline Vector2f operator/(const Vector2f &v2) const;

    [[nodiscard]] __host__ __device__ constexpr inline Vector2f operator*(float t) const noexcept;

    [[nodiscard]] __host__ __device__ constexpr inline Vector2f operator/(float t) const;

    __host__ __device__ constexpr inline Vector2f &operator+=(const Vector2f &v2) noexcept;

    __host__ __device__ constexpr inline Vector2f &operator-=(const Vector2f &v2) noexcept;

    __host__ __device__ constexpr inline Vector2f &operator*=(const Vector2f &v2) noexcept;

    __host__ __device__ constexpr inline Vector2f &operator/=(const Vector2f &v2);

    __host__ __device__ constexpr inline Vector2f &operator*=(float t) noexcept;

    __host__ __device__ constexpr inline Vector2f &operator/=(float t);

    [[nodiscard]] __host__ __device__ constexpr inline float norm() const noexcept;

    [[nodiscard]] __host__ __device__ constexpr inline float squaredNorm() const noexcept;

    [[nodiscard]] __host__ __device__ constexpr inline Vector2f normalized() const;

    [[nodiscard]] __host__ __device__ constexpr inline float dot(const Vector2f &v2) const noexcept;

    __host__ __device__ constexpr inline Vector2f &clamp(float minimum, float maximum) noexcept;


    friend inline std::ostream &operator<<(std::ostream &os, const Vector2f &t);


private:
    float data[2];
};

inline std::ostream &operator<<(std::ostream &os, const Vector2f &t) {
    os << t[0] << ' '
       << t[1] << ' ';
    return os;
}


__host__ __device__ constexpr inline float Vector2f::norm() const noexcept {
    return std::sqrt(squaredNorm());
}

__host__ __device__ constexpr inline float Vector2f::squaredNorm() const noexcept {
    return data[0] * data[0] + data[1] * data[1];
}


__host__ __device__ constexpr inline Vector2f Vector2f::normalized() const {
    float n = norm();
    assert(n != 0);
    float k = 1.f / n;
    return {
            data[0] * k,
            data[1] * k,
    };
}

__host__ __device__ constexpr inline Vector2f Vector2f::operator-() const noexcept {
    return {-data[0], -data[1]};
}


__host__ __device__ constexpr inline Vector2f Vector2f::operator+(const Vector2f &v2) const noexcept {
    return {data[0] + v2.data[0],
            data[1] + v2.data[1]};
}

__host__ __device__ constexpr inline Vector2f Vector2f::operator-(const Vector2f &v2) const noexcept {
    return {data[0] - v2.data[0],
            data[1] - v2.data[1]};
}

__host__ __device__ constexpr inline Vector2f Vector2f::operator*(const Vector2f &v2) const noexcept {
    return {data[0] * v2.data[0],
            data[1] * v2.data[1]};
}

__host__ __device__ constexpr inline Vector2f Vector2f::operator/(const Vector2f &v2) const {
    assert(v2.data[0] != 0);
    assert(v2.data[1] != 0);
    return {data[0] / v2.data[0],
            data[1] / v2.data[1]};
}

__host__ __device__ constexpr inline Vector2f operator*(float t, const Vector2f &v) noexcept {
    return v * t;
}

__host__ __device__ constexpr inline Vector2f Vector2f::operator*(float t) const noexcept {
    return {t * data[0],
            t * data[1]};
}


__host__ __device__ constexpr inline Vector2f Vector2f::operator/(float t) const {
    assert(t != 0);
    return {data[0] / t,
            data[1] / t};
}


__host__ __device__ constexpr inline float Vector2f::dot(const Vector2f &v2) const noexcept {
    return data[0] * v2.data[0] +
           data[1] * v2.data[1];
}

__host__ __device__ constexpr inline Vector2f &Vector2f::operator+=(const Vector2f &v) noexcept {
    data[0] += v.data[0];
    data[1] += v.data[1];
    return *this;
}

__host__ __device__ constexpr inline Vector2f &Vector2f::operator*=(const Vector2f &v) noexcept {
    data[0] *= v.data[0];
    data[1] *= v.data[1];
    return *this;
}

__host__ __device__ constexpr inline Vector2f &Vector2f::operator/=(const Vector2f &v) {
    assert(v.data[0] != 0);
    assert(v.data[1] != 0);

    data[0] /= v.data[0];
    data[1] /= v.data[1];
    return *this;
}

__host__ __device__ constexpr inline Vector2f &Vector2f::operator-=(const Vector2f &v) noexcept {
    data[0] -= v.data[0];
    data[1] -= v.data[1];
    return *this;
}

__host__ __device__ constexpr inline Vector2f &Vector2f::operator*=(float t) noexcept {
    data[0] *= t;
    data[1] *= t;
    return *this;
}

__host__ __device__ constexpr inline Vector2f &Vector2f::operator/=(float t) {
    assert(t != 0);
    float k = 1.f / t;

    data[0] *= k;
    data[1] *= k;
    return *this;
}
__host__ __device__ constexpr inline Vector2f unit_vector(Vector2f v) {
    auto n = v.norm();
    assert(n != 0);
    return v / n;
}

__host__ __device__ constexpr inline Vector2f &Vector2f::clamp(float minimum, float maximum) noexcept {
    data[0] = std::max(minimum, std::min(maximum, data[0]));
    data[1] = std::max(minimum, std::min(maximum, data[1]));
    return *this;
}