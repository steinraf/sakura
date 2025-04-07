cmake_minimum_required(VERSION 3.25.2)
project(CustomRenderer LANGUAGES CXX CUDA)


set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)



set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CUDA_STANDARD 20)





include(FindCUDA/select_compute_arch)
CUDA_DETECT_INSTALLED_GPUS(INSTALLED_GPU_CCS_1)
string(STRIP "${INSTALLED_GPU_CCS_1}" INSTALLED_GPU_CCS_2)
string(REPLACE " " ";" INSTALLED_GPU_CCS_3 "${INSTALLED_GPU_CCS_2}")
string(REPLACE "." "" CUDA_ARCH_LIST "${INSTALLED_GPU_CCS_3}")
SET(CMAKE_CUDA_ARCHITECTURES ${CUDA_ARCH_LIST})

message("COMPILING FOR CUDA_ARCH_LIST: " "${CUDA_ARCH_LIST}")

SET(OIDN_DEVICE_CUDA ON CACHE BOOL "Enable CUDA Device")

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


add_library(pugixml STATIC libs/pugixml/src/pugixml.cpp)

include(CheckLanguage)
check_language(CUDA)
if (CMAKE_CUDA_COMPILER)
    enable_language(CUDA)

    include_directories("${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}")
endif ()

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
        src_old/utility/vector.cuh
        src_old/cudaHelpers.cu
        src_old/scene/scene.cu
        src_old/utility/ray.h
        src_old/hittable.h

        src_old/utility/warp.h
        src_old/utility/sampler.h
        src_old/acceleration/bvh.h
        src_old/shapes/triangle.h
        src_old/utility/meshLoader.cu
        src_old/bsdf.h
        src_old/acceleration/aabb.h
        src_old/scene/sceneLoader.h

        src_old/utility/frame.h
        src_old/textures/imageTexture.cu
        src_old/shapes/triangle.cu
        src_old/denoise/denoise.cu
        src_old/utility/vector.cuh

        src/camera/camera.cuh
        src/common.h
        src/gui/gui.cpp
        src/gui/viewport.cu
        src/statistics/statistics.h
        src/emitters/environmentEmitter.cuh
        src/emitters/areaLight.cu
        src/emitters/queryRecord.cuh
)

target_compile_options(${PROJECT_NAME} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
        -O3
        --extended-lambda
        --expt-relaxed-constexpr
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

add_dependencies(CustomRenderer pugixml)


set_target_properties(CustomRenderer PROPERTIES CUDA_SEPARABLE_COMPILATION ON)



target_link_libraries(CustomRenderer PRIVATE ${CUDA_LIBRARIES} glfw GL GLEW X11 OpenMP::OpenMP_CXX pugixml OpenImageDenoise PNGwriter::PNGwriter tbb ${OPENEXR_LIBRARIES} )

