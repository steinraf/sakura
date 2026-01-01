//
// Created by steinraf on 04.02.25.
//


#include <thrust/device_vector.h>
#include <thrust/sort.h>

#include <utility>

#include "../acceleration/aabb.cuh"
#include "../acceleration/bvh.cuh"
#include "../camera/camera.cuh"

#include "../integrator/integrators.cuh"
#include "scene.cuh"

#include "pugixml.hpp"


void Scene::render(cudaSurfaceObject_t surface, FeatureBuffer *buffer, Camera &camera, curandState *rngStates, const Eigen::Vector2<unsigned int> &windowSize, int spp) const {


    int devId = 0;
    int numSMs;
    checkCudaErrors(cudaDeviceGetAttribute(&numSMs, cudaDevAttrMultiProcessorCount, devId));

    render_kern<<<32 * numSMs, 256>>>(tlas, envmap, buffer, camera, rngStates, windowSize[0], windowSize[1], spp);
    checkCudaErrors(cudaDeviceSynchronize());
    bufferToSurface<<<32 * numSMs, 256>>>(surface, buffer, windowSize[0], windowSize[1]);
    checkCudaErrors(cudaDeviceSynchronize());
}


SceneBuilder &SceneBuilder::addObj(
        const std::string &filename, const Eigen::Affine3f &tf, BSDF bsdf, std::optional<Vec3f> emitterRadiance) {
    std::ifstream file(filename);
    if(!file.is_open()) {
        throw std::runtime_error("Could not open file " + filename);
    }

    std::vector<Vec3f> vertices{};
    std::vector<Vec3f> normals{};
    std::vector<Vec2f> uvs{};

    std::vector<Triangle> trias{};

    std::string lineString{};


    while(std::getline(file, lineString)) {

        std::istringstream line{lineString};
        std::string start;

        line >> start;

        if(start == "v") {
            Vec3f vertex;
            line >> vertex.x() >> vertex.y() >> vertex.z();
            vertices.push_back(vertex);
        } else if(start == "vn") {
            Vec3f normal;
            line >> normal.x() >> normal.y() >> normal.z();
            normals.push_back(normal);
        } else if(start == "vt") {
            Vec2f uv;
            line >> uv.x() >> uv.y();
            uvs.push_back(uv);
        } else if(start == "f") {
            std::array<Vec3f, 3> faceVertices;
            std::array<Vec3f, 3> faceNormals;
            std::array<Vec2f, 3> uvTextures;


            for(int i = 0; i < 3; i++) {
                std::string vertex;
                line >> vertex;
                std::istringstream vertexStream{vertex};
                std::string vertexIndex;
                std::getline(vertexStream, vertexIndex, '/');
                faceVertices.at(i) = vertices.at(std::stoi(vertexIndex) - 1);
                std::string textureIndex;
                std::getline(vertexStream, textureIndex, '/');
                int tIndex = std::stoi(textureIndex);
                if(textureIndex.empty() || tIndex >= uvs.size()) {
                    uvTextures.at(i) = Vec2f{i % 2, i / 2};
                } else {
                    uvTextures.at(i) = uvs.at(tIndex - 1);
                }
                std::string normalIndex;
                std::getline(vertexStream, normalIndex, '/');
                faceNormals.at(i) = normals.at(std::stoi(normalIndex) - 1);
            }


            trias.emplace_back(faceVertices[0], faceVertices[1], faceVertices[2],
                               faceNormals[0], faceNormals[1], faceNormals[2],
                               uvTextures[0], uvTextures[1], uvTextures[2]);

            if(line.peek() == ' ') {
                std::string vertex;
                line >> vertex;
                std::istringstream vertexStream{vertex};
                std::string vertexIndex;
                std::getline(vertexStream, vertexIndex, '/');
                Vec3f faceVertex4 = vertices.at(std::stoi(vertexIndex) - 1);
                std::string textureIndex;
                std::getline(vertexStream, textureIndex, '/');
                int tIndex = std::stoi(textureIndex);
                Vec2f uvTexture4;
                if(textureIndex.empty() || tIndex >= uvs.size()) {
                    uvTexture4 = Vec2f{0, 0};
                } else {
                    uvTexture4 = uvs.at(tIndex - 1);
                }
                std::string normalIndex;
                std::getline(vertexStream, normalIndex, '/');
                auto faceNormal4 = normals.at(std::stoi(normalIndex) - 1);
                trias.emplace_back(faceVertex4, faceVertices[0], faceVertices[2],
                                   faceNormal4, faceNormals[0], faceNormals[2],
                                   uvTexture4, uvTextures[0], uvTextures[2]);
            }
        }
    }

    if(emitterRadiance.has_value()) {
        emitters.push_back({trias, tf, std::move(bsdf), emitterRadiance.value()});
    } else {
        meshes.push_back({trias, tf, std::move(bsdf)});
    }
    return *this;
}


SceneBuilder &SceneBuilder::parseXML(
        const std::string &filename) noexcept(false) {

    // TODO cleanup

    currentXMLRoot = std::filesystem::path{filename}.parent_path();

    auto [doc, root] = initXML(filename);

    //    SceneFactory sceneFactory;
    //    sceneFactory.registerComponent("default", DefaultComponent::create);
    //    //            .registerComponent("sensor", SensorComponent::create);
    //
    //    auto scene = sceneFactory.createScene(root);
    //
    //    return *this;

    [[maybe_unused]] const auto &rootLogger = sceneLogger.getNewSection("scene");

#define CREATE_PARSER(name)                                                                  \
    {                                                                                        \
        #name, [&](const pugi::xml_node &node, auto &logger) { parse_##name(node, logger); } \
    }

    static const std::unordered_map<std::string, std::function<void(const pugi::xml_node &, ScopedLogger &)>>
            parsers{
                    CREATE_PARSER(shape),
                    CREATE_PARSER(sensor),
                    CREATE_PARSER(default),
                    CREATE_PARSER(bsdf),
                    CREATE_PARSER(emitter),
            };


    xmlChildIterator(root, [&](const pugi::xml_node &node) {
        std::string name = node.name();

        auto logger = sceneLogger.getNewSection(name);

        if(auto it = parsers.find(name); it != parsers.end()) {
            it->second(node, logger);
        } else {
            logger.log<true>("Warning: Ignoring XML Node \"" + name + "\"");
        }
    });

    currentXMLRoot.clear();

    return *this;
}

std::pair<pugi::xml_document, pugi::xml_node> SceneBuilder::initXML(const std::string &filename) noexcept(false) {
    pugi::xml_document doc;
    pugi::xml_parse_result result = doc.load_file(filename.c_str());


    // pugixml failed to parse
    if(!result) {
        std::cerr << "XML [" << filename << "] parsed with errors\n";
        std::cerr << "Error description: " << result.description() << "\n";
        std::cerr << "Error offset: " << result.offset << "\n\n";
        throw std::runtime_error("XML parsing error");
    }

    auto root = doc.document_element();


    // The root name must be called "scene"
    if(std::string(root.name()) != "scene") {
        throw std::runtime_error("Unrecognized XML file root in " + filename + '\n');
    }

    return {std::move(doc), root};
}


void SceneBuilder::parse_shape(const pugi::xml_node &shape, auto &logger) {

    auto attribute = lookupName(shape.attribute("type").value());
    if(attribute == "obj") {
        std::filesystem::path objFilename = "";
        Eigen::Affine3f tf = Eigen::Affine3f::Identity();
        BSDF bsdf;

        xmlChildIterator(shape, [&](const pugi::xml_node &node) {
            if(std::string(node.name()) == "string") {
                auto name = lookupName(node.attribute("name").value());
                auto filename = lookupName(node.attribute("value").value());
                if(name != "filename") {
                    throw std::runtime_error("String attribute should have name \"filename\", not " + filename);
                }

                objFilename = currentXMLRoot / filename;

                logger.template log<false>("FOUND OBJ " + filename);
            } else if(std::string(node.name()) == "ref") {
                bsdf = bsdfMap.at(node.attribute("id").value());
                logger.template log<false>(R"(<ref id=")" + lookupName(node.attribute("id").value()) + R"("/>)");
            } else if(std::string(node.name()) == "transform") {
                xmlChildIterator(node, [&](const pugi::xml_node &node) {
                    tf = parseTransform(node, logger.getNewSection("transform"));
                });
            } else if(std::string(node.name()) == "bsdf") {
                auto bsdf_logger = logger.getNewSection("bsdf");
                parse_bsdf(node, bsdf_logger);
            } else {
                logger.template log<true>("Ignoring Shape Node " + std::string(node.name()));
            }
        });

        addObj(objFilename, tf, bsdf);
    } else if(attribute == "rectangle") {
        Eigen::Affine3f tf = Eigen::Affine3f::Identity();
        BSDF bsdf;
        std::optional<Vec3f> emitterRadiance = std::nullopt;

        xmlChildIterator(shape, [&](const pugi::xml_node &node) {
            if(std::string(node.name()) == "ref") {
                bsdf = bsdfMap.at(node.attribute("id").value());
                logger.template log<false>(R"(<ref id=")" + lookupName(node.attribute("id").value()) + R"("/>)");
            } else if(std::string(node.name()) == "transform") {
                xmlChildIterator(node, [&](const pugi::xml_node &node) {
                    tf = parseTransform(node, logger.getNewSection("transform"));
                });
            } else if(std::string(node.name()) == "emitter") {
                xmlChildIterator(node, [&](const pugi::xml_node &node) {
                    if(std::string(node.name()) == "rgb") {
                        if(std::string(node.attribute("name").value()) == "radiance") {
                            emitterRadiance = parseVector(node.attribute("value").value());
                            logger.template log<false>(R"(<rgb name="radiance" value=")" + (std::ostringstream{} << emitterRadiance.value().matrix()).str() + "\"/>");
                        } else {
                            logger.template log<true>("Ignoring RGB " + std::string(node.attribute("name").value()));
                        }
                    } else {
                        logger.template log<true>("Ignoring BSDF " + std::string(node.name()));
                    }
                });
            } else {
                logger.template log<true>("Ignoring Shape Node " + std::string(node.name()));
            }
        });

        if(emitterRadiance.has_value()) {
            addRectangle(tf, bsdf, emitterRadiance);
        } else {
            addRectangle(tf, bsdf);
        }


    } else if(attribute == "cube") {
        Eigen::Affine3f tf = Eigen::Affine3f::Identity();
        BSDF bsdf;

        xmlChildIterator(shape, [&](const pugi::xml_node &node) {
            if(std::string(node.name()) == "ref") {
                bsdf = bsdfMap.at(node.attribute("id").value());
                logger.template log<false>(R"(<ref id=")" + lookupName(node.attribute("id").value()) + R"("/>)");
            } else if(std::string(node.name()) == "transform") {
                xmlChildIterator(node, [&](const pugi::xml_node &node) {
                    tf = parseTransform(node, logger.getNewSection("transform"));
                });
            } else {
                logger.template log<true>("Ignoring Shape Node " + std::string(node.name()));
            }
        });

        addCube(tf, bsdf);

    } else if(attribute == "cube") {
        Eigen::Affine3f tf = Eigen::Affine3f::Identity();
        BSDF bsdf;

        xmlChildIterator(shape, [&](const pugi::xml_node &node) {
            if(std::string(node.name()) == "ref") {
                bsdf = bsdfMap.at(node.attribute("id").value());
                logger.template log<false>(R"(<ref id=")" + lookupName(node.attribute("id").value()) + R"("/>)");
            } else if(std::string(node.name()) == "transform") {
                xmlChildIterator(node, [&](const pugi::xml_node &node) {
                    tf = parseTransform(node, logger.getNewSection("transform"));
                });
            } else {
                logger.template log<true>("Ignoring Shape Node " + std::string(node.name()));
            }
        });

        addCube(tf, bsdf);

    } else if(attribute == "sphere") {
        float radius = 1.0f;
        Vec3f pos = Vec3f::Zero();
        BSDF bsdf;
        std::optional<Vec3f> emitterRadiance = std::nullopt;

        xmlChildIterator(shape, [&](const pugi::xml_node &node) {
            if(std::string(node.name()) == "ref") {
                bsdf = bsdfMap.at(node.attribute("id").value());
                logger.template log<false>(R"(<ref id=")" + lookupName(node.attribute("id").value()) + R"("/>)");
            } else if(std::string(node.name()) == "emitter") {
                xmlChildIterator(node, [&](const pugi::xml_node &node) {
                    if(std::string(node.name()) == "rgb") {
                        if(std::string(node.attribute("name").value()) == "radiance") {
                            emitterRadiance = parseVector(node.attribute("value").value());
                            logger.template log<false>(R"(<rgb name="radiance" value=")" + (std::ostringstream{} << emitterRadiance.value().matrix()).str() + "\"/>");
                        } else {
                            logger.template log<true>("Ignoring RGB " + std::string(node.attribute("name").value()));
                        }
                    } else {
                        logger.template log<true>("Ignoring BSDF " + std::string(node.name()));
                    }
                });
            } else if(std::string(node.name()) == "float") {
                auto name = lookupName(node.attribute("name").value());
                auto value = std::stof(lookupName(node.attribute("value").value()));
                if(name == "radius") {
                    radius = value;
                    logger.template log<false>("<float name=\"radius\" value=\"" + std::to_string(value) + "\"/>");
                } else {
                    logger.template log<true>("Ignoring float attribute " + name);
                }
            } else if(std::string(node.name()) == "point") {
                auto name = lookupName(node.attribute("name").value());
                Vec3f value{
                        node.attribute("x").as_float(),
                        node.attribute("y").as_float(),
                        node.attribute("z").as_float()};
                if(name == "center") {
                    pos = value;
                    logger.template log<false>("<point name=\"center\" value=\"" + (std::ostringstream{} << value.matrix()).str() + "\"/>");
                } else {
                    logger.template log<true>("Ignoring point attribute " + name);
                }

            } else {
                logger.template log<true>("Ignoring Shape Node " + std::string(node.name()));
            }
        });

        if(emitterRadiance.has_value()) {
            addSphere(radius, pos, bsdf, emitterRadiance);
        } else {
            addSphere(radius, pos, bsdf);
        }
    } else {
        logger.template log<true>("Ignoring shape due to attribute " + attribute);
    }
}

void SceneBuilder::parse_emitter(const pugi::xml_node &emitter, auto &logger) {
    auto attribute = lookupName(emitter.attribute("type").value());
    if(attribute == "envmap") {
        std::string filename = "";
        Eigen::Affine3f tf = Eigen::Affine3f::Identity();

        xmlChildIterator(emitter, [&](const pugi::xml_node &node) {
            if(std::string(node.name()) == "string") {
                auto name = lookupName(node.attribute("name").value());
                auto value = lookupName(node.attribute("value").value());
                if(name == "filename") {
                    filename = value;
                    logger.template log<false>("<string name=\"filename\" value=\"" + value + "\"/>");
                } else {
                    logger.template log<true>("Ignoring envmap string attribute " + name);
                }
            } else if(std::string(node.name()) == "transform") {
                xmlChildIterator(node, [&](const pugi::xml_node &node) {
                    tf = parseTransform(node, logger.getNewSection("transform"));
                });
            } else {
                logger.template log<true>("Ignoring Emitter Node " + std::string(node.name()));
            }
        });

        environmentMap = Texture{currentXMLRoot / filename, true, tf};

    } else if(attribute == "directional") {
        Vec3f dir = Vec3f::UnitZ();
        float irradiance = 1.0;
        logger.template log<true>("Replacing directional emitter with far away sphere emitter.");
        xmlChildIterator(emitter, [&](const pugi::xml_node &node) {
            if(std::string(node.name()) == "vector") {
                auto name = lookupName(node.attribute("name").value());
                if(name == "direction") {
                    dir = parseVector(node.attribute("value").value());
                    logger.template log<false>(R"(<vector name="direction" value=")" + (std::ostringstream{} << dir.matrix()).str() + "\"/>");

                } else {
                    logger.template log<true>("Ignoring vector attribute " + name);
                }
            } else if(std::string(node.name()) == "float") {
                auto name = lookupName(node.attribute("name").value());
                if(name == "irradiance") {
                    irradiance = node.attribute("value").as_float();
                    logger.template log<false>(R"(<float name="irradiance" value=")" + std::to_string(irradiance) + "\"/>");

                } else {
                    logger.template log<true>("Ignoring float attribute " + name);
                }
            } else {
                logger.template log<true>("Ignoring Emitter Node " + std::string(node.name()));
            }
        });

        float dist = Texture::EMITTER_DIST() * 0.0001f;
        float r = dist * 0.1f;
        float k = dist * irradiance / (4 * M_PIf);//TODO use correct conversion from irradiance to radiance
        addSphere(r, dist * -dir, BSDF{Material::Diffuse(), Texture::ONES()}, k * Vec3f::Ones());

    } else {
        logger.template log<true>("Ignoring emitter due to attribute " + attribute);
    }
}


Scene SceneBuilder::build() {
    auto scene = Scene{};

    if(meshes.empty()) {
        std::cerr << "No geometry in scene\n";
        throw std::runtime_error("No geometry in scene");
    }
    if(emitters.empty()) {
        std::cout << "WARNING: No emitters in scene\n";
    }

    //TODO cleanup
    TLAS *t;
    checkCudaErrors(cudaMallocManaged(&t, sizeof(TLAS)));
    *t = TLAS(meshes, emitters);

    scene.tlas = t;

    scene.envmap = environmentMap;


    return scene;
}
std::string SceneBuilder::lookupName(const std::string &name) const {
    if(name.empty()) return name;
    if(name[0] != '$') return name;
    try {
        return nameMap.at(name.substr(1));
    } catch(const std::out_of_range &e) {
        throw std::runtime_error("Unknown XML name alias " + name);
    }
}
void SceneBuilder::xmlChildIterator(const pugi::xml_node &node, auto func) const {
    for(const auto &child: node.children()) {
        if(child.type() == pugi::node_comment ||
           child.type() == pugi::node_declaration)
            continue;
        if(child.type() != pugi::node_element)
            throw std::runtime_error("Unknown XML Node encountered.");

        func(child);
    }
}
void SceneBuilder::parse_sensor(const pugi::xml_node &sensor, auto &logger) {
    auto attribute = lookupName(sensor.attribute("type").value());
    if(attribute != "perspective") {
        logger.template log<false>("Non-perspective sensor ignored.");
        return;
    }

    CameraBuilder cameraBuilder;

    Sensor s;

    xmlChildIterator(sensor, [&](const pugi::xml_node &node) {
        if(std::string(node.name()) == "float") {
            auto name = lookupName(node.attribute("name").value());
            auto value = std::stof(lookupName(node.attribute("value").value()));
            if(name == "fov") {
                cameraBuilder.setFOV(value);
                logger.template log<false>(R"(<float name="fov" value=")" + std::to_string(value) + "\"/>");
            } else if(name == "aspectRatio") {
                logger.template log<true>("ASPECT RATIO CAN NOT BE SET MANUALLY");
                //cameraBuilder.setAspectRatio(value);
                //logger.template log<false>("<float name=\"aspectRatio\" value=\"" + std::to_string(value) + "\"/>");
            } else if(name == "aperture") {
                cameraBuilder.setAperture(value);
                logger.template log<false>("<float name=\"aperture\" value=\"" + std::to_string(value) + "\"/>");
            } else if(name == "focusDist") {
                cameraBuilder.setFocusDist(value);
                logger.template log<false>("<float name=\"focusDist\" value=\"" + std::to_string(value) + "\"/>");
            } else if(name == "near") {
                cameraBuilder.setNear(value);
                logger.template log<false>("<float name=\"near\" value=\"" + std::to_string(value) + "\"/>");
            } else if(name == "far") {
                cameraBuilder.setFar(value);
                logger.template log<false>("<float name=\"far\" value=\"" + std::to_string(value) + "\"/>");
            } else {
                logger.template log<true>("Ignoring float attribute " + name);
            }
        } else if(std::string(node.name()) == "transform") {
            Eigen::Isometry3f tf = Eigen::Isometry3f::Identity();
            xmlChildIterator(node, [&](const pugi::xml_node &node) {
                tf = parseTransform(node, logger.getNewSection("transform"));
            });
            cameraBuilder.setTransform(tf);
        } else if(std::string(node.name()) == "sampler") {


            if(std::string(node.attribute("type").value()) == "independent") {
                xmlChildIterator(node, [&](const pugi::xml_node &node) {
                    if(std::string(node.name()) == "integer") {
                        auto name = lookupName(node.attribute("name").value());
                        auto value = std::stoi(lookupName(node.attribute("value").value()));
                        if(name == "sample_count") {
                            s.samplingPattern.spp = value;
                            logger.template log<false>("<integer name=\"sample_count\" value=\"" + std::to_string(value) + "\"/>");
                        } else {
                            logger.template log<true>("Ignoring integer attribute " + name);
                        }
                    } else {
                        logger.template log<true>("Ignoring XML Node " + std::string(node.name()));
                    }
                });
            } else {
                logger.template log<false>("Ignoring sampler type " + std::string(node.attribute("type").value()) + ". Using independent.");
            }
        } else if(std::string(node.name()) == "film") {
            if(std::string(node.attribute("type").value()) == "hdrfilm") {
                xmlChildIterator(node, [&](const pugi::xml_node &node) {
                    if(std::string(node.name()) == "integer") {
                        auto name = lookupName(node.attribute("name").value());
                        auto value = std::stoi(lookupName(node.attribute("value").value()));
                        if(name == "width") {
                            s.film.size[0] = value;
                            logger.template log<false>("<integer name=\"width\" value=\"" + std::to_string(value) + "\"/>");
                        } else if(name == "height") {
                            s.film.size[1] = value;
                            logger.template log<false>("<integer name=\"height\" value=\"" + std::to_string(value) + "\"/>");
                        } else {
                            logger.template log<true>("Ignoring integer attribute " + name);
                        }
                    } else {
                        logger.template log<true>("Ignoring XML Node " + std::string(node.name()));
                    }
                });
            } else {
                logger.template log<false>("Ignoring film type " + std::string(node.attribute("type").value()) + ". Using hdrfilm.");
            }
        } else {
            logger.template log<true>("Ignoring XML Node " + std::string(node.name()));
        }
    });

    cameraBuilder.setAspectRatio(float(s.film.size[0]) / float(s.film.size[1]));


    s.camera = cameraBuilder.build();

    logger.template log<false>("=> AspectRatio: " + std::to_string(float(s.film.size[0]) / float(s.film.size[1])));


    sensors.push_back(s);
}
Eigen::Isometry3f SceneBuilder::parseTransform(const pugi::xml_node &node, auto logger) const {
    auto nodeName = std::string(node.name());
    Eigen::Isometry3f tf = Eigen::Isometry3f::Identity();
    auto vecToString = [](const Vec3f &vec) {
        return (std::ostringstream{} << vec[0] << ' ' << vec[1] << ' ' << vec[2]).str();
    };
    if(std::string(node.name()) == "matrix") {
        std::string matrix = node.attribute("value").value();
        std::replace(matrix.begin(), matrix.end(), ',', ' ');
        std::istringstream matrixStream{matrix};
        for(int i = 0; i < 4; i++) {
            for(int j = 0; j < 4; j++) {
                matrixStream >> tf.matrix()(i, j);
            }
        }
        if(std::string tmp; matrixStream >> tmp) {
            throw std::runtime_error("Too many values in vector " + matrix);
        }
        logger.template log<false>("<matrix value=\"" + (std::ostringstream{} << tf.matrix()).str() + "\"/>");
    } else if(std::string(node.name()) == "lookat") {
        Vec3f target = -Vec3f::UnitZ(), origin = Vec3f::Zero(), up = Vec3f::UnitY();
        for(const auto &attribute: node.attributes()) {
            if(std::string(attribute.name()) == "target") {
                target = parseVector(attribute.value());
                logger.template log<false>("<lookat target=\"" + vecToString(target) + "\"/>");
            } else if(std::string(attribute.name()) == "origin") {
                origin = parseVector(attribute.value());
                logger.template log<false>("<lookat origin=\"" + vecToString(origin) + "\"/>");
            } else if(std::string(attribute.name()) == "up") {
                up = parseVector(attribute.value());
                logger.template log<false>("<lookat up=\"" + vecToString(up) + "\"/>");
            } else {
                logger.template log<true>("Ignoring attribute " + std::string(attribute.name()));
            }
        }
        tf = Camera::lookAt(origin, target, up);
    } else if(std::string(node.name()) == "rotate") {
        // <rotate x="1" angle="114"/>
        Vec3f axis = [&node]() {
            if(node.find_attribute([](const pugi::xml_attribute &attr) { return std::string(attr.name()) == "x"; })) {
                return Vec3f::UnitX();
            } else if(node.find_attribute([](const pugi::xml_attribute &attr) { return std::string(attr.name()) == "y"; })) {
                return Vec3f::UnitY();
            } else if(node.find_attribute([](const pugi::xml_attribute &attr) { return std::string(attr.name()) == "z"; })) {
                return Vec3f::UnitZ();
            } else {
                throw std::runtime_error("No axis specified in rotate");
            }
        }();
        tf.rotate(Eigen::AngleAxisf(std::stof(node.attribute("angle").value()), axis));
    } else if(std::string(node.name()) == "lookat") {
        Vec3f target = -Vec3f::UnitZ(), origin = Vec3f::Zero(), up = Vec3f::UnitY();
        for(const auto &attribute: node.attributes()) {
            if(std::string(attribute.name()) == "target") {
                target = parseVector(attribute.value());
                logger.template log<false>("<lookat target=\"" + vecToString(target) + "\"/>");
            } else if(std::string(attribute.name()) == "origin") {
                origin = parseVector(attribute.value());
                logger.template log<false>("<lookat origin=\"" + vecToString(origin) + "\"/>");
            } else if(std::string(attribute.name()) == "up") {
                up = parseVector(attribute.value());
                logger.template log<false>("<lookat up=\"" + vecToString(up) + "\"/>");
            } else {
                logger.template log<true>("Ignoring attribute " + std::string(attribute.name()));
            }
        }
        tf = Camera::lookAt(origin, target, up);
    } else {
        logger.template log<true>("Unknown transform type " + std::string(node.name()));
    }


    return tf;
}
Vec3f SceneBuilder::parseVector(std::string str) {
    std::replace(str.begin(), str.end(), ',', ' ');
    std::istringstream stream{str};
    Vec3f vec;
    if(!(stream >> vec.x() >> vec.y() >> vec.z())) {
        throw std::runtime_error("Could not parse vector " + str);
    }

    if(std::string tmp; stream >> tmp) {
        throw std::runtime_error("Too many values in vector " + str);
    }
    return vec;
}
SceneBuilder &SceneBuilder::getSensors(std::vector<Sensor> &s) {
    s.resize(this->sensors.size());
    for(unsigned int i = 0; i < s.size(); i++) {
        s[i] = this->sensors[i];
    }
    return *this;
}
void SceneBuilder::parse_default(const pugi::xml_node &node, auto &logger) {
    unsigned int attrCount = 0;
    for(const auto &attribute: node.attributes()) {
        attrCount++;
    }

    if(attrCount == 2) {
        auto name = lookupName(node.attribute("name").value());
        auto value = lookupName(node.attribute("value").value());
        nameMap[name] = value;
        logger.template log<false>("name=\"" + name + "\" value=\"" + value + "\"");
    } else {
        logger.template log<true>("Too many attributes in default " + std::string(node.name()) + ". Ignoring.");
    }
}
void SceneBuilder::parse_bsdf(const pugi::xml_node &bsdf, auto &logger) {
    std::string id = bsdf.attribute("id").value();
    logger.template log<false>("id=\"" + id + "\"");
    if(std::string(bsdf.attribute("type").value()) == "twosided") {
        logger.template log<false>(R"(type="twosided")");
        xmlChildIterator(bsdf, [&](const pugi::xml_node &node) {
            if(std::string(node.name()) == "bsdf") {
                if(std::string(node.attribute("type").value()) == "diffuse") {
                    xmlChildIterator(node, [&](const pugi::xml_node &node) {
                        if(std::string(node.name()) == "rgb") {
                            if(std::string(node.attribute("name").value()) == "reflectance") {
                                Vec3f color = parseVector(node.attribute("value").value());
                                bsdfMap[id] = BSDF{Material{MaterialType::DIFFUSE}, Texture{color}};
                                logger.template log<false>(R"(<rgb name="reflectance" value=")" + (std::ostringstream{} << color.matrix()).str() + "\"/>");
                            } else {
                                logger.template log<true>("Ignoring RGB " + std::string(node.attribute("name").value()));
                            }
                        } else if(std::string(node.name()) == "texture") {
                            if(std::string(node.attribute("name").value()) == "reflectance") {
                                xmlChildIterator(node, [&](const pugi::xml_node &node) {
                                    if(std::string(node.name()) == "string") {
                                        auto name = lookupName(node.attribute("name").value());
                                        auto filename = lookupName(node.attribute("value").value());
                                        if(name == "filename") {
                                            bsdfMap[id] = BSDF{Material{MaterialType::DIFFUSE}, Texture{currentXMLRoot / filename}};
                                            logger.template log<false>("<string name=\"filename\" value=\"" + filename + "\"/>");
                                        } else if(name == "filter_type") {
                                            if(lookupName(node.attribute("value").value()) == "bilinear") {
                                                logger.template log<false>("<string name=\"filter_type\" value=\"bilinear\"/>");
                                            } else {
                                                logger.template log<true>("Ignoring String " + std::string(node.attribute("name").value()));
                                            }
                                        } else {
                                            logger.template log<true>("Ignoring String " + std::string(node.attribute("name").value()));
                                        }

                                    } else {
                                        logger.template log<true>("Ignoring Texture " + std::string(node.name()));
                                    }
                                });
                            } else {
                                logger.template log<true>("Ignoring Texture " + std::string(node.attribute("name").value()));
                            }
                        } else {
                            logger.template log<true>("Ignoring BSDF " + std::string(node.name()));
                        }
                    });
                } else if(std::string(node.attribute("type").value()) == "roughconductor") {


                    logger.template log<false>(R"(type="roughconductor")");
                    logger.template log<true>("Converting roughconductor BSDF to regular conductor BSDF");
                    bsdfMap[id] = BSDF{Material{MaterialType::SPECULAR}, Texture::DEFAULT()};


                    //                    logger.template log<true>("Converting roughconductor BSDF to microfacet BSDF");
                    //                    float alpha = node.child("float").attribute("value").as_float();
                    //                    float intIOR = 1.0f, extIOR = 1.5f;
                    //                    Color kd = parseVector(node.find_child_by_attribute("rgb", "name", "specular_reflectance").attribute("value").value());
                    //                    bsdfMap[id] = BSDF{Material{
                    //                                               MaterialType::MICROFACET,
                    //                                               alpha,
                    //                                               1.5f, 1.0f, kd},
                    //                                       Texture::DEFAULT()};
                    //                    logger.template log<false>(R"(<float name="alpha" value=")" + std::to_string(alpha) + "\"/>");
                    //                    logger.template log<false>(R"(<float name="int_ior" value=")" + std::to_string(intIOR) + "\"/>");
                    //                    logger.template log<false>(R"(<float name="ext_ior" value=")" + std::to_string(extIOR) + "\"/>");
                    //                    logger.template log<false>(R"(<rgb name="specular_reflectance" value=")" + (std::ostringstream{} << kd.matrix()).str() + "\"/>");


                } else if(std::string(node.attribute("type").value()) == "coating") {


                    logger.template log<false>(R"(type="coating")");
                    logger.template log<true>("Converting coating BSDF to regular conductor BSDF");
                    bsdfMap[id] = BSDF{Material{MaterialType::SPECULAR}, Texture::DEFAULT()};


                    //                    logger.template log<true>("Converting roughconductor BSDF to microfacet BSDF");
                    //                    float alpha = node.child("float").attribute("value").as_float();
                    //                    float intIOR = 1.0f, extIOR = 1.5f;
                    //                    Color kd = parseVector(node.find_child_by_attribute("rgb", "name", "specular_reflectance").attribute("value").value());
                    //                    bsdfMap[id] = BSDF{Material{
                    //                                               MaterialType::MICROFACET,
                    //                                               alpha,
                    //                                               1.5f, 1.0f, kd},
                    //                                       Texture::DEFAULT()};
                    //                    logger.template log<false>(R"(<float name="alpha" value=")" + std::to_string(alpha) + "\"/>");
                    //                    logger.template log<false>(R"(<float name="int_ior" value=")" + std::to_string(intIOR) + "\"/>");
                    //                    logger.template log<false>(R"(<float name="ext_ior" value=")" + std::to_string(extIOR) + "\"/>");
                    //                    logger.template log<false>(R"(<rgb name="specular_reflectance" value=")" + (std::ostringstream{} << kd.matrix()).str() + "\"/>");


                } else if(std::string(node.attribute("type").value()) == "conductor") {
                    logger.template log<false>(R"(type="conductor")");
                    bsdfMap[id] = BSDF{Material{MaterialType::SPECULAR}, Texture::DEFAULT()};
                } else if(std::string(node.attribute("type").value()) == "plastic") {
                    logger.template log<false>(R"(type="plastic")");
                    logger.template log<true>("Converting plastic BSDF to regular diffuse BSDF");
                    xmlChildIterator(node, [&](const pugi::xml_node &node) {
                        if(std::string(node.name()) == "rgb") {
                            if(std::string(node.attribute("name").value()) == "diffuse_reflectance") {
                                Vec3f color = parseVector(node.attribute("value").value());
                                bsdfMap[id] = BSDF{Material{MaterialType::DIFFUSE}, Texture{color}};
                                logger.template log<false>(R"(<rgb name="diffuse_reflectance" value=")" + (std::ostringstream{} << color.matrix()).str() + "\"/>");
                            } else {
                                logger.template log<true>("Ignoring RGB " + std::string(node.attribute("name").value()));
                            }
                        } else {
                            logger.template log<true>("Ignoring BSDF " + std::string(node.name()));
                        }
                    });
                } else if(std::string(node.attribute("type").value()) == "roughplastic") {
                    logger.template log<false>(R"(type="roughplastic")");
                    logger.template log<true>("Converting roughplastic BSDF to regular diffuse BSDF");
                    xmlChildIterator(node, [&](const pugi::xml_node &node) {
                        if(std::string(node.name()) == "rgb") {
                            if(std::string(node.attribute("name").value()) == "diffuse_reflectance") {
                                Vec3f color = parseVector(node.attribute("value").value());
                                bsdfMap[id] = BSDF{Material{MaterialType::DIFFUSE}, Texture{color}};
                                logger.template log<false>(R"(<rgb name="diffuse_reflectance" value=")" + (std::ostringstream{} << color.matrix()).str() + "\"/>");
                            } else {
                                logger.template log<true>("Ignoring RGB " + std::string(node.attribute("name").value()));
                            }
                        } else {
                            logger.template log<true>("Ignoring BSDF " + std::string(node.name()));
                        }
                    });
                } else {
                    logger.template log<true>("Ignoring BSDF with type " + std::string(node.attribute("type").value()));
                }
            } else {
                logger.template log<true>("Invalid BSDF Child Node " + std::string(node.name()));
            }
        });
    } else if(std::string(bsdf.attribute("type").value()) == "bumpmap") {
        logger.template log<false>("Recursively parsing bumpmap bsdf");
        auto bsdfLoggerSection = logger.getNewSection("bsdf");
        parse_bsdf(bsdf.find_child([](const pugi::xml_node &node) { return std::string(node.name()) == "bsdf"; }), bsdfLoggerSection);
    } else if(std::string(bsdf.attribute("type").value()) == "dielectric") {
        float intIOR = 1.5f, extIOR = 1.0f;
        xmlChildIterator(bsdf, [&](const pugi::xml_node &node) {
            if(std::string(node.name()) == "float") {
                auto name = lookupName(node.attribute("name").value());
                auto value = std::stof(lookupName(node.attribute("value").value()));
                if(name == "int_ior" || name == "intIOR") {
                    intIOR = value;
                    logger.template log<false>(R"(<float name="int_ior" value=")" + std::to_string(value) + "\"/>");
                } else if(name == "ext_ior" || name == "extIOR") {
                    extIOR = value;
                    logger.template log<false>(R"(<float name="ext_ior" value=")" + std::to_string(value) + "\"/>");
                } else {
                    logger.template log<true>("Ignoring float attribute " + name);
                }
            } else {
                logger.template log<true>("Ignoring XML Node " + std::string(node.name()));
            }
        });
        bsdfMap[id] = BSDF{Material{MaterialType::DIELECTRIC, intIOR, extIOR}, Texture::DEFAULT()};
    } else if(std::string(bsdf.attribute("type").value()) == "thindielectric") {
        logger.template log<false>(R"(type="thindielectric")");
        logger.template log<true>("Converting thindielectric BSDF to regular dielectric BSDF");
        float intIOR = 1.5f, extIOR = 1.0f;
        xmlChildIterator(bsdf, [&](const pugi::xml_node &node) {
            if(std::string(node.name()) == "float") {
                auto name = lookupName(node.attribute("name").value());
                auto value = std::stof(lookupName(node.attribute("value").value()));
                if(name == "int_ior") {
                    intIOR = value;
                    logger.template log<false>(R"(<float name="int_ior" value=")" + std::to_string(value) + "\"/>");
                } else if(name == "ext_ior") {
                    extIOR = value;
                    logger.template log<false>(R"(<float name="ext_ior" value=")" + std::to_string(value) + "\"/>");
                } else {
                    logger.template log<true>("Ignoring float attribute " + name);
                }
            } else {
                logger.template log<true>("Ignoring XML Node " + std::string(node.name()));
            }
        });
        bsdfMap[id] = BSDF{Material{MaterialType::DIELECTRIC, intIOR, extIOR}, Texture::DEFAULT()};
    } else {
        logger.template log<true>("Ignoring BSDF " + std::string(bsdf.attribute("type").value()));
    }

    if(id.empty()) {
        logger.template log<true>("Warning: BSDF without id");
    } else if(!bsdfMap.contains(id)) {
        logger.template log<true>("Substituting BSDF with default diffuse");
        bsdfMap[id] = BSDF{Material{MaterialType::DIFFUSE}, Texture::DEFAULT()};
    }
}
SceneBuilder &SceneBuilder::addRectangle(const Eigen::Affine3f &tf, BSDF bsdf, std::optional<Vec3f> emitterRadiance) {
    std::vector<Triangle> trias{
            Triangle{
                    {-1, -1, 0},
                    {1, -1, 0},
                    {1, 1, 0},
                    {0, 0, 1},
                    {0, 0, 1},
                    {0, 0, 1},
                    {0, 0},
                    {1, 0},
                    {1, 1},
            },
            Triangle{{-1, -1, 0}, {1, 1, 0}, {-1, 1, 0}, {0, 0, 1}, {0, 0, 1}, {0, 0, 1}, {0, 0}, {1, 1}, {0, 1}},
    };

    if(emitterRadiance.has_value()) {
        emitters.push_back({trias, tf, std::move(bsdf), emitterRadiance.value()});
    } else {
        meshes.push_back({trias, tf, std::move(bsdf)});
    }

    return *this;
}

SceneBuilder &SceneBuilder::addCube(const Eigen::Affine3f &tf, BSDF bsdf, std::optional<Vec3f> emitterRadiance) {
    std::vector<Triangle> trias{
            Triangle{{-1, -1, -1}, {1, -1, -1}, {1, 1, -1}, {0, 0, -1}, {0, 0, -1}, {0, 0, -1}, {0, 0}, {1, 0}, {1, 1}},
            Triangle{{-1, -1, -1}, {1, 1, -1}, {-1, 1, -1}, {0, 0, -1}, {0, 0, -1}, {0, 0, -1}, {0, 0}, {1, 1}, {0, 1}},
            Triangle{{-1, -1, 1}, {1, -1, 1}, {1, 1, 1}, {0, 0, 1}, {0, 0, 1}, {0, 0, 1}, {0, 0}, {1, 0}, {1, 1}},
            Triangle{{-1, -1, 1}, {1, 1, 1}, {-1, 1, 1}, {0, 0, 1}, {0, 0, 1}, {0, 0, 1}, {0, 0}, {1, 1}, {0, 1}},
            Triangle{{-1, -1, -1}, {-1, 1, -1}, {-1, 1, 1}, {-1, 0, 0}, {-1, 0, 0}, {-1, 0, 0}, {0, 0}, {1, 0}, {1, 1}},
            Triangle{{-1, -1, -1}, {-1, 1, 1}, {-1, -1, 1}, {-1, 0, 0}, {-1, 0, 0}, {-1, 0, 0}, {0, 0}, {1, 1}, {0, 1}},
            Triangle{{1, -1, -1}, {1, 1, -1}, {1, 1, 1}, {1, 0, 0}, {1, 0, 0}, {1, 0, 0}, {0, 0}, {1, 0}, {1, 1}},
            Triangle{{1, -1, -1}, {1, 1, 1}, {1, -1, 1}, {1, 0, 0}, {1, 0, 0}, {1, 0, 0}, {0, 0}, {1, 1}, {0, 1}},
            Triangle{{-1, -1, -1}, {1, -1, -1}, {1, -1, 1}, {0, -1, 0}, {0, -1, 0}, {0, -1, 0}, {0, 0}, {1, 0}, {1, 1}},
            Triangle{{-1, -1, -1}, {1, -1, 1}, {-1, -1, 1}, {0, -1, 0}, {0, -1, 0}, {0, -1, 0}, {0, 0}, {1, 1}, {0, 1}},
            Triangle{{-1, 1, -1}, {1, 1, -1}, {1, 1, 1}, {0, 1, 0}, {0, 1, 0}, {0, 1, 0}, {0, 0}, {1, 0}, {1, 1}},
            Triangle{{-1, 1, -1}, {1, 1, 1}, {-1, 1, 1}, {0, 1, 0}, {0, 1, 0}, {0, 1, 0}, {0, 0}, {1, 1}, {0, 1}},
    };


    if(emitterRadiance.has_value()) {
        throw std::runtime_error("Emitter Radiance not supported for OBJs");
        emitters.push_back({trias, tf, std::move(bsdf), emitterRadiance.value()});
    } else {
        meshes.push_back({trias, tf, std::move(bsdf)});
    }

    return *this;
}
SceneBuilder &SceneBuilder::addSphere(float radius, const Vec3f &center, BSDF bsdf, std::optional<Vec3f> emitterRadiance) {
    Eigen::Affine3f tf = Eigen::Affine3f::Identity();
    tf.translate(center);
    tf.scale(Vec3f::Constant(radius));

    addObj("scenes/sphere.obj", tf, std::move(bsdf), std::move(emitterRadiance));

    return *this;
}


ScopedLogger::~ScopedLogger() {
    formatter.dedent();
    log<false>("</" + tagName + ">");
}
ScopedLogger::ScopedLogger(SceneLogger &formatter, std::string tagName, const std::string &attribute) : formatter(formatter), tagName(std::move(tagName)) {
    log<false>('<' + this->tagName + (attribute.empty() ? "" : " ") + attribute + '>');
    formatter.indent();
}
ScopedLogger ScopedLogger::getNewSection(std::string name, const std::string &attribute) {
    return formatter.getNewSection(std::move(name), attribute);
}

template<bool isError>
void ScopedLogger::log(const std::string &msg) const {
    assert(formatter.indentLevel >= 0);
    // Create initial indent
    std::string indentedMsg = std::string(formatter.indentLevel, '\t') + msg;

    // In case there are line breaks, indent them as well
    size_t pos;
    do {
        pos = indentedMsg.find('\n');

        if constexpr(isError) {
            const static std::string orange = "\033[38;2;255;165;0m";
            const static std::string reset = "\033[0m";
            std::cout << orange << indentedMsg.substr(0, pos) << reset << std::endl;
        } else {
            std::cout << indentedMsg.substr(0, pos) << std::endl;
        }
        indentedMsg = std::string(formatter.indentLevel, '\t') + indentedMsg.substr(pos + 1);
    } while(pos != std::string::npos);
}

ScopedLogger SceneLogger::getNewSection(std::string tagName, const std::string &attribute) {
    return ScopedLogger{*this, std::move(tagName), attribute};
}

SceneFactory &SceneFactory::registerComponent(const std::string &name, ComponentGenerator generator) {

    if(componentGenerators.contains(name)) {
        throw std::runtime_error("Component " + name + " already registered");
    }

    componentGenerators[name] = std::move(generator);

    return *this;
}
std::unique_ptr<Component> SceneFactory::createComponent(const pugi::xml_node &node) noexcept(false) {
    auto name = node.name();
    if(!componentGenerators.contains(name)) {
        throw std::runtime_error("Unknown component " + std::string(name));
    }

    auto componentRequest = [this](const std::string &name) { return requestComponent(name); };
    auto componentAdder = [this](const std::string &name, std::unique_ptr<Component> component) { addComponent(name, std::move(component)); };


    auto component = componentGenerators[name](node, componentRequest, componentAdder);
    return component;
}
std::shared_ptr<Component> SceneFactory::requestComponent(const std::string &name) const noexcept(false) {
    if(!componentMap.contains(name)) {
        throw std::runtime_error("Unknown component " + name);
    }

    return componentMap.at(name);
}
void SceneFactory::addComponent(const std::string &name, std::unique_ptr<Component> component) {
    if(componentMap.contains(name)) {
        throw std::runtime_error("Component " + name + " already added");
    }

    componentMap[name] = std::move(component);
}
SceneDescriptor SceneFactory::createScene(const pugi::xml_node &root) noexcept(false) {
    SceneDescriptor sceneDescriptor;
    for(const auto &child: root.children()) {
        if(child.type() == pugi::node_comment ||
           child.type() == pugi::node_declaration)
            continue;
        if(child.type() != pugi::node_element)
            throw std::runtime_error("Unknown XML Node encountered.");

        sceneDescriptor.components.push_back(createComponent(child));
    }
    return sceneDescriptor;
}
std::unique_ptr<Component> DefaultComponent::create(const pugi::xml_node &node, ComponentRequestor requestor, ComponentAdder adder) noexcept(false) {
    return std::make_unique<DefaultComponent>();
}
