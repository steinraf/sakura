#pragma once


#include <Eigen/Dense>
#include <chrono>
#include <fstream>
#include <iostream>
#include <map>
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
    [[nodiscard]] AABB getBoundingBox() const { return tlas->getBoundingBox(); };

private:
    friend class SceneBuilder;
    Scene() = default;
    TLAS *tlas;
    Texture envmap;
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

struct ComponentIdea {
    // Example
    /*
     * <[name] ([attribute0=value0], ...)>
     *  (
     *   [child0],
     *   ...
     *  )
     * </[name]>
     */
    std::string name;
    struct Attribute {
        std::string name;
        std::string value;
        bool isRequired;
    };
    std::vector<Attribute> attributes;
    struct Child {
        std::string name;
        bool isRequired;
    };
    std::vector<Child> children;
};

struct SceneComponent;
using Component = SceneComponent;
using ComponentRequestor = std::function<std::shared_ptr<Component>(const std::string &name)>;
using ComponentAdder = std::function<void(const std::string &name, std::unique_ptr<Component>)>;
using ComponentGenerator = std::function<std::unique_ptr<Component>(const pugi::xml_node &node, ComponentRequestor requestor, ComponentAdder adder)>;

struct SceneComponent {
    virtual ~SceneComponent() = default;
};

struct DefaultComponent : public SceneComponent {
public:
    static std::unique_ptr<Component> create(const pugi::xml_node &node, ComponentRequestor requestor, ComponentAdder adder) noexcept(false);

private:
    std::string name;
    std::string value;
};

struct SceneDescriptor {
    std::vector<std::unique_ptr<Component>> components;
};

class SceneFactory {

public:
    SceneFactory &registerComponent(const std::string &name, ComponentGenerator generator);

    std::unique_ptr<Component> createComponent(const pugi::xml_node &node) noexcept(false);
    SceneDescriptor createScene(const pugi::xml_node &root) noexcept(false);

private:
    friend Component;
    std::shared_ptr<Component> requestComponent(const std::string &name) const noexcept(false);
    void addComponent(const std::string &name, std::unique_ptr<Component> component);

    std::map<std::string, ComponentGenerator> componentGenerators;
    std::map<std::string, std::shared_ptr<Component>> componentMap;
};


class SceneBuilder {
public:
    SceneBuilder() = default;


    // Load scene component from file
    SceneBuilder &parseXML(const std::string &filename) noexcept(false);
    SceneBuilder &addObj(const std::string &filename, const Eigen::Affine3f &tf = Eigen::Affine3f::Identity(), BSDF bsdf = {}, std::optional<Vec3f> emitterRadiance = std::nullopt);
    SceneBuilder &addRectangle(const Eigen::Affine3f &tf, BSDF bsdf = {}, std::optional<Vec3f> emitterRadiance = std::nullopt);
    SceneBuilder &addCube(const Eigen::Affine3f &tf, BSDF bsdf = {}, std::optional<Vec3f> emitterRadiance = std::nullopt);
    SceneBuilder &addSphere(float radius, const Vec3f &center, BSDF bsdf = {}, std::optional<Vec3f> emitterRadiance = std::nullopt);

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
    void parse_bsdf(const pugi::xml_node &bsdf, auto &logger);
    void parse_emitter(const pugi::xml_node &emitter, auto &logger);

    // Prepare the XML file for parsing
    // returns document and document root
    [[nodiscard]] static std::pair<pugi::xml_document, pugi::xml_node> initXML(const std::string &filename) noexcept(false);

    void xmlChildIterator(const pugi::xml_node &node, auto func) const;

    Eigen::Isometry3f parseTransform(const pugi::xml_node &node, auto logger) const;
    static Eigen::Vector3f parseVector(std::string str);

    [[nodiscard]] std::string lookupName(const std::string &name) const;
    std::unordered_map<std::string, std::string> nameMap;
    std::unordered_map<std::string, BSDF> bsdfMap;

    std::vector<MeshDescriptorHost> meshes;
    std::vector<EmitterDescriptorHost> emitters;
    std::vector<Sensor> sensors;

    Texture environmentMap = Texture::ZERO();

    SceneLogger sceneLogger;

    std::filesystem::path currentXMLRoot;
};
