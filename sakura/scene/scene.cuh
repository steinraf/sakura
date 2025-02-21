#pragma once


#include <Eigen/Dense>
#include <chrono>
#include <fstream>
#include <iostream>
#include <optional>
#include <vector>

#include <curand_kernel.h>
#include <omp.h>

#include "imgui.h"
#include "pngwriter.h"
#include "pugixml.hpp"

#include "../common.cuh"
#include "../geometry/triangle.cuh"


class Scene {

public:
    void render(cudaSurfaceObject_t surface, FeatureBuffer *buffer, Camera &camera, curandState *rngStates, const ImVec2 &windowSize, int spp) const;

private:
    friend class SceneBuilder;
    Scene() = default;
    BVH *bvh;
};

class SceneLogger;

struct ScopedLogger {
public:
    // Constructor prints <tagName attribute>
    //                    \t[...]
    // Destructor prints  </tagName>

    [[nodiscard]] ScopedLogger getNewSection(std::string tagName, std::string attribute = "");

    // Automatically indents and ends the line
    // If isError, the message is printed in orange
    template<bool isError>
    void log(const std::string &msg) const;

    explicit ScopedLogger(SceneLogger &formatter, std::string tagName, std::string attribute);
    ScopedLogger() = delete;
    ScopedLogger &operator=(const ScopedLogger &) = delete;
    ScopedLogger(const ScopedLogger &) = delete;

    ~ScopedLogger();


private:
    class SceneLogger &formatter;
    std::string tagName;
};

class SceneLogger {
public:
    [[nodiscard]] ScopedLogger getNewSection(std::string tagName, std::string attribute = "");

private:
    void indent() { ++indentLevel; }
    void dedent() { --indentLevel; }


    friend ScopedLogger;

    int indentLevel = 0;
};

class SceneBuilder {
public:
    explicit SceneBuilder() = default;


    // Load scene component from file
    SceneBuilder &parseXML(const std::string &filename) noexcept(false);
    SceneBuilder &addObj(const std::string &filename, const Eigen::Affine3f &tf = Eigen::Affine3f::Identity());

    // Directly add Components
    SceneBuilder &addTriangle(const Triangle &triangle);


    [[nodiscard]] Scene build();


private:
    //parse_n function to shape xml element n
    //has to match upper/lower case because of macro


    void parse_shape(const pugi::xml_node &shape, const auto &logger);


    // Prepare the XML file for parsing
    // returns document and document root
    [[nodiscard]] std::pair<pugi::xml_document, pugi::xml_node> loadXML(const std::string &filename) const noexcept(false);

    [[nodiscard]] std::string lookupName(const std::string &name) const;
    std::unordered_map<std::string, std::string> nameMap;

    std::vector<Triangle> triangles;

    SceneLogger sceneLogger{};
};


__device__ __host__ constexpr uint32_t LeftShift3(uint32_t x) noexcept;