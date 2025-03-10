#pragma once


#include <Eigen/Dense>
#include <chrono>
#include <fstream>
#include <iostream>
#include <optional>
#include <vector>

#include <curand_kernel.h>
#include <filesystem>
#include <omp.h>

#include "imgui.h"
#include "pngwriter.h"
#include "pugixml.hpp"

#include "../acceleration/multibvh.cuh"
#include "../common.cuh"
#include "../geometry/triangle.cuh"


class Scene {

public:
    void render(cudaSurfaceObject_t surface, FeatureBuffer *buffer, Camera &camera, curandState *rngStates, const Eigen::Vector2<unsigned int> &windowSize, int spp) const;

private:
    friend class SceneBuilder;
    Scene() = default;
    TLAS *tlas;
};

class SceneLogger;

struct ScopedLogger {
public:
    // Constructor prints <name attribute>
    //                    \t[...]
    // Destructor prints  </name>
    [[nodiscard]] ScopedLogger getNewSection(std::string name, const std::string &attribute = "");

    // Automatically indents and ends the line
    // If isError, the message is printed in orange
    template<bool isError>
    void log(const std::string &msg) const;

    explicit ScopedLogger(SceneLogger &formatter, std::string tagName, const std::string &attribute);
    ScopedLogger() = delete;
    ScopedLogger &operator=(const ScopedLogger &) = delete;
    ScopedLogger(const ScopedLogger &) = delete;

    ~ScopedLogger();


private:
    class SceneLogger &formatter;
    std::string tagName; /* Name of the tag */
};

class SceneLogger {
public:
    [[nodiscard]] ScopedLogger getNewSection(std::string tagName, const std::string &attribute = "");

private:
    void indent() { ++indentLevel; }
    void dedent() { --indentLevel; }


    friend ScopedLogger;

    int indentLevel = 0;
};

class SceneBuilder {
public:
    SceneBuilder() = default;


    // Load scene component from file
    SceneBuilder &parseXML(const std::string &filename) noexcept(false);
    SceneBuilder &addObj(const std::string &filename, const Eigen::Affine3f &tf = Eigen::Affine3f::Identity(), Material material = Material(), Texture texture = Texture());

    // Directly add Components
    //    SceneBuilder &addTriangle(const Triangle &triangle);

    SceneBuilder &getSensors(std::vector<Sensor> &s);


    [[nodiscard]] Scene build();


private:
    //parse_n function to shape xml element n
    //has to match upper/lower case because of macro


    void parse_shape(const pugi::xml_node &shape, auto &logger);
    void parse_sensor(const pugi::xml_node &sensor, auto &logger);
    void parse_default(const pugi::xml_node &node, auto &logger);

    // Prepare the XML file for parsing
    // returns document and document root
    [[nodiscard]] static std::pair<pugi::xml_document, pugi::xml_node> loadXML(const std::string &filename) noexcept(false);

    void xmlChildIterator(const pugi::xml_node &node, auto func) const;

    Eigen::Isometry3f parseTransform(const pugi::xml_node &node, auto logger) const;
    Eigen::Vector3f parseVector(std::string str) const;

    [[nodiscard]] std::string lookupName(const std::string &name) const;
    std::unordered_map<std::string, std::string> nameMap;

    std::vector<MeshDescriptorHost> meshes;
    std::vector<Sensor> sensors;

    SceneLogger sceneLogger;

    std::filesystem::path currentXMLRoot;
};
