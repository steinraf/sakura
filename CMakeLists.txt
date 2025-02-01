cmake_minimum_required(VERSION 3.25.2)
project(CustomRenderer LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

#include(FindCUDA/select_compute_arch)
#CUDA_DETECT_INSTALLED_GPUS(INSTALLED_GPU_CCS_1)
#string(STRIP "${INSTALLED_GPU_CCS_1}" INSTALLED_GPU_CCS_2)
#string(REPLACE " " ";" INSTALLED_GPU_CCS_3 "${INSTALLED_GPU_CCS_2}")
#string(REPLACE "." "" CUDA_ARCH_LIST "${INSTALLED_GPU_CCS_3}")
#SET(CMAKE_CUDA_ARCHITECTURES ${CUDA_ARCH_LIST})
#


set(OIDN_NEURAL_RUNTIME DNNL)

find_package(TBB REQUIRED)



include_directories(libs/pugixml/src)
include_directories(libs/oneTBB)
include_directories(libs/oneTBB/include)

add_subdirectory(libs/OpenImageDenoise)

include_directories(libs/OpenImageDenoise/include)
include_directories(libs/ImGui)
include_directories(libs/ImGui/backends)
include_directories(libs/pngwriter/src)
include_directories(libs/libpng/)
include_directories(libs/libpng/scripts/)
include_directories(libs/zlib)
include_directories(libs/freetype/include)

include_directories(libs/glew-2.1.0/include)





add_library(pugixml STATIC libs/pugixml/src/pugixml.cpp)


include_directories(libs/stb)
include_directories(libs/ImGui)
include_directories(libs/ImGui/backends)
include_directories(libs/glfw)
include_directories(libs/glfw/include)



find_package(PNGwriter 0.7.0 REQUIRED)
#find_package(glfw REQUIRED)
find_package(OpenGL REQUIRED)
find_package(X11 REQUIRED)
find_package(OpenMP REQUIRED)

include_directories(${CMAKE_SOURCE_DIR}/includes)

#set(CMAKE_CXX_FLAGS "-O3 -fopenmp -funroll-loops -DNO_FREETYPE -D_GLIBCXX_ASSERTIONS -fasynchronous-unwind-tables -fexceptions -fpie -Wl,-pie -fstack-clash-protection -fstack-protector-all -fstack-protector-strong -g -grecord-gcc-switches -fcf-protection -pipe -Wall -Werror -Wextra ")

set(CMAKE_CUDA_FLAGS "-extended-lambda --expt-relaxed-constexpr -dopt=on -extra-device-vectorization --restrict --compiler-options -Wall")


add_executable(CustomRenderer

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
        src/utility/vector.h
        src/cudaHelpers.cu
        src/cudaHelpers.h
        src/scene/scene.cu
        src/scene/scene.h
        src/utility/ray.h
        src/hittable.h
        src/camera/camera.cu
        src/camera/camera.h
        src/utility/warp.h
        src/utility/sampler.h
        src/acceleration/bvh.h
        src/shapes/triangle.h
        src/utility/meshLoader.h
        src/utility/meshLoader.cu
        src/bsdf.h
        src/acceleration/aabb.h
        src/scene/sceneLoader.h
        src/emitters/areaLight.h
        src/emitters/areaLight.cu
        src/utility/frame.h
        src/textures/imageTexture.cu
        src/textures/imageTexture.h
        src/emitters/environmentEmitter.h
        src/shapes/triangle.cu
        src/medium/medium.h
        src/medium/medium.cu
        src/medium/isotropicPhaseFunction.cu
        src/medium/isotropicPhaseFunction.h
        src/denoise/denoise.cu
        src/denoise/denoise.h)

add_dependencies(CustomRenderer pugixml)


set_target_properties(CustomRenderer PROPERTIES CUDA_SEPARABLE_COMPILATION ON)



target_link_libraries(CustomRenderer PRIVATE glfw GL GLEW X11 pthread pugixml OpenImageDenoise PNGwriter::PNGwriter TBB ${OPENEXR_LIBRARIES})

