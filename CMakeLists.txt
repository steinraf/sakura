cmake_minimum_required(VERSION 3.25.2)
project(Sakura LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)


set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CUDA_STANDARD 20)


find_package(Eigen3 REQUIRED)


include(FindCUDA/select_compute_arch)
CUDA_DETECT_INSTALLED_GPUS(INSTALLED_GPU_CCS_1)
string(STRIP "${INSTALLED_GPU_CCS_1}" INSTALLED_GPU_CCS_2)
string(REPLACE " " ";" INSTALLED_GPU_CCS_3 "${INSTALLED_GPU_CCS_2}")
string(REPLACE "." "" CUDA_ARCH_LIST "${INSTALLED_GPU_CCS_3}")
SET(CMAKE_CUDA_ARCHITECTURES ${CUDA_ARCH_LIST})

message("COMPILING FOR CUDA_ARCH_LIST: " "${CUDA_ARCH_LIST}")

set(OIDN_NEURAL_RUNTIME DNNL)

find_package(TBB REQUIRED)


include_directories(libs/pugixml/src)
include_directories(libs/oneTBB)
include_directories(libs/oneTBB/include)


include_directories(libs/ImGui)
include_directories(libs/ImGui/backends)
include_directories(libs/pngwriter/src)
include_directories(libs/libpng/)
include_directories(libs/libpng/scripts/)
include_directories(libs/zlib)
include_directories(libs/freetype/include)

include_directories(libs/glew-2.1.0/include)


find_program(WAYLAND_SCANNER_EXECUTABLE NAMES wayland-scanner)
if (NOT WAYLAND_SCANNER_EXECUTABLE)
    set(GLFW_BUILD_WAYLAND False)
endif ()


include(FetchContent)

#find_package(glfw3 REQUIRED)
# Download and build GLFW
FetchContent_Declare(
        glfw
        GIT_REPOSITORY https://github.com/glfw/glfw.git
        GIT_TAG 3.4
)
FetchContent_MakeAvailable(glfw)


include_directories(libs/stb)
include_directories(libs/ImGui)
include_directories(libs/ImGui/backends)
include_directories(libs/glfw)
include_directories(libs/glfw/include)


add_library(pugixml STATIC libs/pugixml/src/pugixml.cpp)


find_package(PNGwriter 0.7.0 REQUIRED)
find_package(OpenGL REQUIRED)
find_package(X11 REQUIRED)
find_package(OpenMP REQUIRED)

include_directories(${CMAKE_SOURCE_DIR}/includes)
include_directories(${OPENGL_INCLUDE_DIRS})
#include_directories("${CUDA_INCLUDE_DIRS}")

#find_package(CUDA REQUIRED)
include(CheckLanguage)
check_language(CUDA)
if (CMAKE_CUDA_COMPILER)
    enable_language(CUDA)

    # to tell your code it was found via #ifdef USE_CUDA:
    add_definitions(-DUSE_CUDA)

    include_directories("${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}")
endif ()


add_executable(${PROJECT_NAME}

        libs/ImGui/imgui.cpp
        libs/ImGui/imgui_demo.cpp
        libs/ImGui/imgui_draw.cpp
        libs/ImGui/imgui_internal.h
        libs/ImGui/imgui_tables.cpp
        libs/ImGui/imgui_widgets.cpp

        libs/pngwriter/src/pngwriter.cc
        libs/pngwriter/src/pngwriter.h

        libs/ImGui/backends/imgui_impl_glfw.cpp
        libs/ImGui/backends/imgui_impl_opengl3.cpp


        main.cu
        sakura/acceleration/aabb.cu
        sakura/acceleration/bvh.cu
        sakura/acceleration/multibvh.cu
        sakura/camera/camera.cu
        sakura/common.cu
        sakura/denoise/denoise.cu
        sakura/geometry/triangle.cu
        sakura/geometry/intersection.cuh
        sakura/geometry/ray.cu
        sakura/gui/gui.cu
        sakura/gui/viewport.cu
        sakura/integrator/integrators.cu
        sakura/material/bsdf.cu
        sakura/material/material.cu
        sakura/rng/sampler.cu
        sakura/scene/scene.cu
        sakura/texture/texture.cu
        sakura/emitter/emitter.cu
)


target_compile_options(${PROJECT_NAME} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
        -O3
        --extended-lambda
        --expt-relaxed-constexpr
        -diag-suppress 20012
        -fopenmp
        -funroll-loops
        -DNO_FREETYPE
        -D_GLIBCXX_ASSERTIONS
        -fasynchronous-unwind-tables
        -fexceptions
        -fpie
        -Wl,-pie
        -fstack-clash-protection
        -fstack-protector-all
        -fstack-protector-strong
        -g
        -grecord-gcc-switches
        -fcf-protection
        -pipe
        -Wall
        >)

set_property(TARGET ${PROJECT_NAME} PROPERTY "${CUDA_ARCH_LIST}")

add_dependencies(${PROJECT_NAME} pugixml)


target_compile_features(${PROJECT_NAME} PUBLIC cxx_std_20)
set_target_properties(${PROJECT_NAME} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)


target_link_libraries(${PROJECT_NAME} PRIVATE ${CUDA_LIBRARIES} ${OPENGL_LIBRARIES} glfw GL GLEW X11 pthread pugixml PNGwriter::PNGwriter ${OPENEXR_LIBRARIES} Eigen3::Eigen)

