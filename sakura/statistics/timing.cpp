//
// Created by steinraf on 12.02.25.
//

#include "timing.h"
Timer::Timer(VoidFunction auto &&func) {
    auto start = std::chrono::high_resolution_clock::now();
    func();
    checkCudaErrors(cudaDeviceSynchronize());
    auto end = std::chrono::high_resolution_clock::now();
    duration = end - start;
}
double Timer::getDuration() const {
    return duration.count();
}
