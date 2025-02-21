//
// Created by steinraf on 12.02.25.
//

#pragma once


#include <chrono>
#include <concepts>

#include <cuda.h>
#include <cuda_runtime.h>

#include "../common.cuh"

template<typename F>
concept VoidFunction = requires(F f) {
    { f() } -> std::same_as<void>;
};

class Timer {
public:
    Timer() = delete;
    explicit Timer(VoidFunction auto &&func);

    [[nodiscard]] double getDuration() const;

private:
    std::chrono::duration<double> duration;
};