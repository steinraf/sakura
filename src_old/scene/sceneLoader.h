//
// Created by steinraf on 19/11/22.
//

#pragma once

#include <cassert>
#include <filesystem>
#include <iostream>
#include <map>
#include <string>

#include "../../src/camera/camera.cuh"
#include "../bsdf.h"
#include "../utility/vector.cuh"
#include "pugixml.hpp"


struct SceneRepresentation {
    explicit SceneRepresentation(const std::filesystem::path &file) noexcept(false) {
        pugi::xml_document doc;
        pugi::xml_parse_result result = doc.load_file(file.c_str());


        if(!result) {
            std::cout << "XML [" << file.string() << "] parsed with errors\n";
            std::cout << "Error description: " << result.description() << "\n";
            std::cout << "Error offset: " << result.offset << "\n\n";
        }

        auto root = doc.document_element();

        std::cout << "Started parsing " << root.name() << '\n';

        if(std::string(root.name()) != "scene") {
            throw std::runtime_error("Unrecognized XML file root\n");
        }

        for(const auto &node: root.children()) {

            if(node.type() == pugi::node_comment || node.type() == pugi::node_declaration)
                continue;
            if(node.type() != pugi::node_element)
                throw std::runtime_error("Unknown XML Node encounted.");

            const std::string &name = node.name();
            std::cout << '\t' << name;
            if(node.attribute("id"))
                std::cout << ": " << getString(node.attribute("id"));
            std::cout << '\n';


            if(name == "default") {
                addDefault(node);
            } else if(name == "integrator") {
                setIntegrator(node);
            } else if(name == "sensor") {
                setSensor(node);
            } else if(name == "shape") {
                if(node.child("emitter")) {
                    addEmitter(node);
                } else {
                    addMesh(node);
                }
            } else if (name == "emitter") {
                addEnvironmentEmitter(node);
            }else {
                    throw std::runtime_error("Unrecognized node name \"" + name + "\" while parsing XML.");
            }
        }
    }

    void inline addDefault(const pugi::xml_node &node) noexcept {
        map[node.attribute("name").value()] = node.attribute("value").value();
        std::cout << "\t\t"
                  << "Added mapping " << node.attribute("name").value() << " -> "
                  << node.attribute("value").value() << '\n';
    }

    void inline setIntegrator(const pugi::xml_node &node) noexcept(false) {
        if(getString(node.attribute("type")) != "path")
            throw std::runtime_error("Integrator must be path, not \"" + getString(node.attribute("type")) + "\".");
        std::cout << "\t\t path\n";

        for(const auto &child: node.children()) {
            const std::string &childName = child.name();
            if(childName == "integer") {
                if(getString(child.attribute("name")) != "max_depth")
                    throw std::runtime_error(
                            "Unrecognized integrator integer option " + getString(child.attribute("name")));

                sceneInfo.maxRayDepth = std::stof(getString(child.attribute("value")));
                std::cout << "\t\tMaxRayDepth: \n\t\t\t" << sceneInfo.maxRayDepth << '\n';
            } else {
                throw std::runtime_error("Non-Integer child found in integrator.");
            }
        }
    }

    void inline setSensor(const pugi::xml_node &node) noexcept(false) {
        if(getString(node.attribute("type")) != "perspective")
            throw std::runtime_error("Sensor must be perspective, not \"" + getString(node.attribute("type")) + "\" .");

        for(const auto &child: node.children()) {
            const std::string &childName = child.name();
            if(childName == "float") {
                auto attribName = getString(child.attribute("name"));
                if(attribName == "fov") {
                    cameraInfo.fov = std::stof(getString(child.attribute("value")));
                    std::cout << "\t\tFOV: \n\t\t\t" << cameraInfo.fov << '\n';
                } else if(attribName == "aperture_radius") {
                    cameraInfo.aperture = std::stof(getString(child.attribute("value")));
                    std::cout << "\t\tAperture: \n\t\t\t" << cameraInfo.aperture << '\n';
                } else if(attribName == "focus_distance"){
                    cameraInfo.focusDist = std::stof(getString(child.attribute("value")));
                    std::cout << "\t\tAperture: \n\t\t\t" << cameraInfo.focusDist << '\n';
                } else if(attribName == "k1"){
                    cameraInfo.k1 = std::stof(getString(child.attribute("value")));
                    std::cout << "\t\tK1: \n\t\t\t" << cameraInfo.k1 << '\n';
                } else if(attribName == "k2"){
                    cameraInfo.k2 = std::stof(getString(child.attribute("value")));
                    std::cout << "\t\tK2: \n\t\t\t" << cameraInfo.k2 << '\n';
                } else {
                    throw std::runtime_error("Unrecognized sensor float option " + getString(child.attribute("name")));
                }
            } else if(childName == "transform") {
                std::cout << "\t\t"
                          << "Transform: \n";
                auto lookAt = child.child("lookat");
                if(lookAt) {
                    cameraInfo.target = getVector3f(lookAt, "target", "\t\t\t");
                    cameraInfo.origin = getVector3f(lookAt, "origin", "\t\t\t");
                    cameraInfo.up = getVector3f(lookAt, "up", "\t\t\t");
                } else {
                    throw std::runtime_error("Sensor Transform must contain lookAt.");
                }

            } else if(childName == "sampler") {

                std::cout << "\t\t"
                          << "Sampler: " << '\n';

                if(getString(child.attribute("type")) != "independent")
                    throw std::runtime_error(
                            "Sampler type must be independent, not \"" + getString(child.attribute("type")) + "\".");

                auto sampleCount = child.child("integer");
                if(getString(sampleCount.attribute("name")) != "sample_count")
                    throw std::runtime_error(
                            "Encountered unknown attribute \"" + getString(sampleCount.attribute("name")) +
                            "\" in sampler.");
                sceneInfo.samplePerPixel = std::stoi(getString(sampleCount.attribute("value")));
                std::cout << "\t\t\t"
                          << "SampleCount: " << sceneInfo.samplePerPixel << '\n';
            } else if(childName == "film") {

                if(getString(child.attribute("type")) != "hdrfilm")
                    throw std::runtime_error(
                            "Invalid Film type \"" + getString(child.attribute("type")) + "\" encountered.");

                for(const auto &filmChild: child.children()) {
                    const std::string &filmChildName = filmChild.name();
                    if(filmChildName == "rfilter") {
                        std::cerr << "WARNING, IGNORING FILTERS\n";
                    } else if(filmChildName == "integer") {
                        if(getString(filmChild.attribute("name")) == "width") {
                            sceneInfo.width = std::stoi(getString(filmChild.attribute("value")));
                            std::cout << "\t\t\t"
                                      << "Width: " << sceneInfo.width << '\n';
                        } else if(getString(filmChild.attribute("name")) == "height") {
                            sceneInfo.height = std::stoi(getString(filmChild.attribute("value")));
                            std::cout << "\t\t\t"
                                      << "Height: " << sceneInfo.height << '\n';
                        } else {
                            throw std::runtime_error(
                                    "Unknown integer option \"" + getString(filmChild.attribute("name")) +
                                    "\"for hdrfilm encountered.");
                        }
                    } else {
                        throw std::runtime_error("Unknown option \"" + filmChildName + "\" for hdrfilm encountered.");
                    }
                }
            } else {
                throw std::runtime_error("Invalid child tag \"" + childName + "\" found for sensor.");
            }
        }
    }

    void inline addFilename(const std::string &name, bool isEmitter = false) {
        if(isEmitter) {
            emitterInfos.back().filename = name;
            std::cout << "\t\tFilename: " << emitterInfos.back().filename << '\n';
        } else {
            meshInfos.back().filename = name;
            std::cout << "\t\tFilename: " << meshInfos.back().filename << '\n';
        }
    }

    void inline addTextureFilename(const std::string &name, bool isEmitter = false) {
        if(isEmitter) {
            emitterInfos.back().textureName = name;
            std::cout << "\t\tTexture: " << emitterInfos.back().textureName << '\n';
        } else {
            meshInfos.back().textureName = name;
            std::cout << "\t\tTexture: " << meshInfos.back().textureName << '\n';
        }
    }

    void addBSDF(const pugi::xml_node &node, bool isEmitter = false) {
        const std::string attribName = getString(node.attribute("type"));
        std::cout << "\t\tMaterial:\n";
        if(attribName == "diffuse") {
            const auto color = node.child("rgb");
            assert(color ||
                   (isEmitter && !emitterInfos.back().textureName.empty()) ||
                   (!isEmitter && !meshInfos.back().textureName.empty()));
            std::cout << "\t\t\tBSDF: DIFFUSE\n";
            if(isEmitter) {
                if(emitterInfos.back().textureName.empty()) {
                    if(getString(color.attribute("name")) != "reflectance")
                        throw std::runtime_error(
                                "Invalid Tag \"" + getString(color.attribute("name")) + "\"found for material.");
                    emitterInfos.back().bsdf = {Material::DIFFUSE, getVector3f(color, "value", "\t\t\t", "reflectance")};

                } else {
                    emitterInfos.back().bsdf = {Material::DIFFUSE, emitterInfos.back().textureName};
                }
            } else {
                if(meshInfos.back().textureName.empty()) {
                    if(getString(color.attribute("name")) != "reflectance")
                        throw std::runtime_error(
                                "Invalid Tag \"" + getString(color.attribute("name")) + "\"found for material.");

                    meshInfos.back().bsdf = {Material::DIFFUSE, getVector3f(color, "value", "\t\t\t", "reflectance")};
                } else {
                    meshInfos.back().bsdf = {Material::DIFFUSE, meshInfos.back().textureName};
                }
            }
        } else if(attribName == "mirror") {
            std::cout << "\t\t\tBSDF: MIRROR\n";
            if(isEmitter) {
                emitterInfos.back().bsdf = {Material::MIRROR, Color3f{0.f}};
            } else {
                meshInfos.back().bsdf = {Material::MIRROR, Color3f{0.f}};
            }
        } else if(attribName == "dielectric") {
            std::cout << "\t\t\tBSDF: DIELECTRIC\n";
            if(isEmitter) {
                emitterInfos.back().bsdf = {Material::DIELECTRIC, Color3f{0.f}};
            } else {
                meshInfos.back().bsdf = {Material::DIELECTRIC, Color3f{0.f}};
            }
        } else {
            throw std::runtime_error("Invalid Material \"" + getString(node.attribute("type")) + "\".");
        }
    }

    void inline addNormalMap(const std::string &name, bool isEmitter = false) {
        if(isEmitter) {
            emitterInfos.back().normalMap = Texture{name};
            std::cout << "\t\tNormal Map: " << name << '\n';
        } else {
            meshInfos.back().normalMap = Texture{name};
            std::cout << "\t\tNormal Map: " << name << '\n';
        }
    }

    void inline addMesh(const pugi::xml_node &node) noexcept(false) {
        if(getString(node.attribute("type")) != "obj")
            throw std::runtime_error("Error while parsing shape. Only .obj files are supported.");

        meshInfos.emplace_back();

        for(const auto &child: node.children()) {
            const std::string &childName = child.name();
            if(childName == "string") {
                if(getString(child.attribute("name")) == "filename") {
                    addFilename(getString(child.attribute("value")));
                } else if(getString(child.attribute("name")) == "texture") {
                    addTextureFilename(getString(child.attribute("value")));
                } else if(getString(child.attribute("name")) == "normal_map") {
                    addNormalMap(getString(child.attribute("value")));
                } else {
                    throw std::runtime_error(
                            "Unknown mesh string option \"" + getString(child.attribute("name")) + "\" found.");
                }
            } else if(childName == "bsdf") {
                addBSDF(child);
            } else if(childName == "transform") {
                createTransform(child);
            }else {
                throw std::runtime_error("Invalid Tag \"" + childName + "\" found for shape.");
            }
        }
    }

    void inline addEmitter(const pugi::xml_node &node) noexcept(false) {
        if(getString(node.attribute("type")) != "obj")
            throw std::runtime_error("Error while parsing shape. Only .obj files are supported.");

        emitterInfos.emplace_back();

        for(const auto &child: node.children()) {
            const std::string &childName = child.name();

            if(childName == "string") {
                if(getString(child.attribute("name")) == "filename") {
                    addFilename(getString(child.attribute("value")), true);
                } else if(getString(child.attribute("name")) == "texture") {
                    addTextureFilename(getString(child.attribute("value")), true);
                } else if(getString(child.attribute("name")) == "normal_map") {
                    addNormalMap(getString(child.attribute("value")), true);
                }
            } else if(childName == "bsdf") {
                addBSDF(child, true);
            } else if(childName == "emitter") {
                if(getString(child.attribute("type")) != "area")
                    throw std::runtime_error(
                            "Invalid Emitter \"" + getString(child.attribute("type")) + "\". Only area is supported.");

                const auto color = child.child("rgb");
                if(getString(color.attribute("name")) != "radiance")
                    throw std::runtime_error(
                            "Invalid Tag \"" + getString(color.attribute("name")) + "\" found for emitter.");

                std::cout << "\t\tEmitter:\n";
                std::cout << "\t\t\tType: Area\n";

                emitterInfos.back().radiance = getVector3f(color, "value", "\t\t\t", "radiance");

            } else if(childName == "transform") {
                createTransform(child, true);
            } else {
                throw std::runtime_error("Invalid Tag \"" + childName + "\" found for shape.");
            }
        }
    }

    void inline addEnvironmentEmitter(const pugi::xml_node &node){
        if(getString(node.attribute("type")) != "envmap")
            throw std::runtime_error("Error while parsing emitter. Only envmap type is supported.");

        for(const auto &child: node.children()) {
            const std::string &childName = child.name();
            if(childName == "string") {
                if(getString(child.attribute("name")) == "filename") {
                    environmentInfo.texture = Texture{getString(child.attribute("value"))};
                } else if (getString(child.attribute("name")) == "constant") {
                    environmentInfo.texture = Texture{getVector3f(child, "value", "\t\t\t", "texture")};

                }else {
                        throw std::runtime_error("Unknown envmap string option \"" + getString(child.attribute("name")) + "\" found.");
                }
            }else{
                throw std::runtime_error("Unknown envmap child \"" + childName + "\" found.");
            }

        }

    }


    [[nodiscard]] Vector3f inline getVector3f(pugi::xml_node node, const std::string &name, const std::string &indents = "",
                                              const std::string &outName = "") const noexcept {
        auto attrib = node.attribute(name.c_str());
        const std::string targetS = getString(attrib);
        std::cout << indents << (outName.empty() ? name : outName) << ": {" << targetS << "}\n";

        return Vector3f{targetS};
    }

    [[nodiscard]] std::string getString(const pugi::xml_attribute &attrib) const noexcept {
        if(attrib.value()[0] == '$')
            return map.at(std::string(attrib.value()).substr(1));

        return attrib.value();
    }


    void inline createTransform(const pugi::xml_node &transform, bool isEmitter = false) noexcept(false) {


        Matrix4f &currentTransform = [&]() -> Matrix4f & {
            if(isEmitter)
                return emitterInfos.back().transform;
            else
                return meshInfos.back().transform;
        }();


        std::cout << "\t\tTransform: \n";
        for(auto &child: transform.children()) {
            const std::string &tfChildName = child.name();
            if(tfChildName == "translate") {
                currentTransform = Matrix4f(getVector3f(child, "value", "\t\t\t", "translation")) * currentTransform;
            } else if(tfChildName == "scale") {
                currentTransform =
                        Matrix4f(Matrix3f::fromDiag(getVector3f(child, "value", "\t\t\t", "scale"))) * currentTransform;
            } else if(tfChildName == "rotateAxis") {
                const float angle = std::stof(child.attribute("angle").value());
                currentTransform = Matrix4f({getVector3f(child, "axis", "\t\t\t", "rotation axis"),
                                             angle}) *
                                   currentTransform;
                std::cout << "\t\t\trotation angle: " << angle << "°\n";
            } else if(tfChildName == "matrix") {
                throw std::runtime_error("Matrix transforms are not implemented yet.");
                //TODO implement matrix transform
            } else {
                throw std::runtime_error("Invalid Tag \"" + tfChildName + "\" found for transform");
            }
        }
    }

    struct CameraInfo {
        CameraInfo()
            : target(0.f, 0.f, -1.f),
              origin(0.f), up(0.f, 1.f, 0.f),
              fov(30), aperture(0.f), focusDist(1.f),
              k1(0.f), k2(0.f), apertureType(ApertureType::Circular) {
        }

        Vector3f target, origin, up;
        float fov, aperture, focusDist;
        float k1, k2;
        ApertureType apertureType;
    };

    CameraInfo cameraInfo;


    struct MeshInfo {
        MeshInfo() = default;

        std::string filename;
        std::string textureName;
        Matrix4f transform;
        BSDF bsdf;
//        Texture normalMap = Texture{"scenes/normalMaps/rock.jpg"};
        Texture normalMap = Texture{Vector3f{0.5f, 0.5f, 1.f}};
    };

    std::vector<MeshInfo> meshInfos{};

    struct EmitterInfo {
        EmitterInfo() = default;

        std::string filename;
        std::string textureName;
        Matrix4f transform;
        BSDF bsdf;
        Vector3f radiance;
        Texture normalMap = Texture{Vector3f{0.5f, 0.5f, 1.f}};
    };

    std::vector<EmitterInfo> emitterInfos{};

    struct SceneInfo {
        SceneInfo()
            : samplePerPixel(4), width(100), height(100), maxRayDepth(6) {
        }

        int samplePerPixel;
        int width, height;
        int maxRayDepth;
    };

    SceneInfo sceneInfo;

    struct EnvironmentInfo{
        Texture texture{Color3f{0.f, 0.f, 0.f}};
    };

    EnvironmentInfo environmentInfo;

private:
    std::unordered_map<std::string, std::string> map;
};
