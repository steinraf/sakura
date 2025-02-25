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

    render_kern<<<32 * numSMs, 256>>>(bvh, buffer, camera, rngStates, windowSize[0], windowSize[1], spp);
    checkCudaErrors(cudaDeviceSynchronize());
    bufferToSurface<<<32 * numSMs, 256>>>(surface, buffer, windowSize[0], windowSize[1]);
    checkCudaErrors(cudaDeviceSynchronize());
}


SceneBuilder &SceneBuilder::addObj(
        const std::string &filename, const Eigen::Affine3f &tf) {
    std::ifstream file(filename);
    if(!file.is_open()) {
        throw std::runtime_error("Could not open file " + filename);
    }

    std::vector<Eigen::Vector3f> vertices{};
    std::vector<Eigen::Vector3f> normals{};

    std::vector<Triangle> trias{};

    std::string lineString{};

    while(std::getline(file, lineString)) {
        std::istringstream line{lineString};
        std::string start;

        line >> start;

        if(start == "v") {
            Eigen::Vector3f vertex;
            line >> vertex.x() >> vertex.y() >> vertex.z();
            vertices.push_back(vertex);
        } else if(start == "vn") {
            Eigen::Vector3f normal;
            line >> normal.x() >> normal.y() >> normal.z();
            normals.push_back(normal);
        } else if(start == "f") {
            std::array<Eigen::Vector3f, 3> faceVertices;
            std::array<Eigen::Vector3f, 3> faceNormals;

            for(int i = 0; i < 3; i++) {
                std::string vertex;
                line >> vertex;
                std::istringstream vertexStream{vertex};
                std::string vertexIndex;
                std::getline(vertexStream, vertexIndex, '/');
                faceVertices[i] = vertices[std::stoi(vertexIndex) - 1];
                std::string textureIndex;
                std::getline(vertexStream, textureIndex, '/');
                //                uvTextures[i] = uvs[std::stoi(textureIndex) -
                //                1];
                std::string normalIndex;
                std::getline(vertexStream, normalIndex, '/');
                faceNormals[i] = normals[std::stoi(normalIndex) - 1];
            }


            trias.emplace_back(
                    tf * faceVertices[0], tf * faceVertices[1],
                    tf * faceVertices[2], tf.linear() * faceNormals[0],
                    tf.linear() * faceNormals[1], tf.linear() * faceNormals[2]);
        }
    }
    for(const auto &t: trias) {
        addTriangle(t);
    }
    return *this;
}

SceneBuilder &SceneBuilder::addTriangle(const Triangle &triangle) {
    triangles.push_back(triangle);
    return *this;
}


SceneBuilder &SceneBuilder::parseXML(
        const std::string &filename) noexcept(false) {

    currentXMLRoot = std::filesystem::path{filename}.parent_path();

    auto [doc, root] = loadXML(filename);

    [[maybe_unused]] const auto &rootLogger = sceneLogger.getNewSection("scene");

#define CREATE_PARSER(name) {#name, [&](const pugi::xml_node &node, auto &logger) { parse_##name(node, logger); }}

    static const std::unordered_map<std::string, std::function<void(const pugi::xml_node &, ScopedLogger &)>>
            parsers{
                    CREATE_PARSER(shape),
                    CREATE_PARSER(sensor),
                    CREATE_PARSER(default),
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

std::pair<pugi::xml_document, pugi::xml_node> SceneBuilder::loadXML(const std::string &filename) noexcept(false) {
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

    // version string used in mitsuba is not supported here
    if(root.attribute("version")) {
        std::cout << "Warning: Ignoring scene version string " << root.attribute("version").value() << '\n';
    }

    // The root name must be called "scene"
    if(std::string(root.name()) != "scene") {
        throw std::runtime_error("Unrecognized XML file root in " + filename + '\n');
    }

    return {std::move(doc), root};
}


void SceneBuilder::parse_shape(const pugi::xml_node &shape, auto &logger) {

    auto attribute = lookupName(shape.attribute("type").value());
    if(attribute == "obj") {

        xmlChildIterator(shape, [&](const pugi::xml_node &node) {
            if(std::string(node.name()) == "string") {
                auto name = lookupName(node.attribute("name").value());
                auto filename = lookupName(node.attribute("value").value());
                if(name != "filename") {
                    throw std::runtime_error("String attribute should have name \"filename\", not " + filename);
                }

                addObj(currentXMLRoot / filename);
                logger.template log<false>("FOUND OBJ " + filename);
            } else {
                logger.template log<true>("Ignoring XML Node " + std::string(node.name()));
            }
        });


    } else {
        logger.template log<true>("Ignoring shape due to attribute " + attribute);
    }
}


Scene SceneBuilder::build() {
    auto scene = Scene{};

    if(triangles.empty()) {
        std::cerr << "No geometry in scene\n";
        throw std::runtime_error("No geometry in scene");
    }

    Triangle *trias;
    checkCudaErrors(
            cudaMallocManaged(&trias, triangles.size() * sizeof(Triangle)));
    checkCudaErrors(cudaMemcpy(trias, triangles.data(),
                               triangles.size() * sizeof(Triangle),
                               cudaMemcpyHostToDevice));

    std::cout << "Allocated Triangles\n";

    AABB boundingBox = thrust::transform_reduce(
            thrust::device, trias, trias + triangles.size(),
            [=] __host__ __device__(const Triangle &t) -> AABB {
                return t.AABBGetter();
            },
            AABB{}, thrust::plus<AABB>());

    std::cout << "Scene Bounding Box: Min\n"
              << boundingBox.min << "\nMax\n"
              << boundingBox.max << '\n';

    Eigen::Vector3f lower = boundingBox.min;
    Eigen::Vector3f dims = boundingBox.max - boundingBox.min;

    thrust::device_vector<uint32_t> mortonCodes(triangles.size());
    thrust::transform(
            thrust::device, trias, trias + triangles.size(), mortonCodes.begin(),
            [=] __host__ __device__(const Triangle &tria) {
                int numBits = 10;
                const Eigen::Vector3f normalized =
                        static_cast<float>(1u << numBits) *
                        (tria.AABBGetter().getCenter() - lower).array() / dims.array();

                assert(normalized[0] >= 0 && normalized[1] >= 0 &&
                       normalized[2] >= 0);
                assert(normalized[0] <= (1u << numBits) &&
                       normalized[1] <= (1u << numBits) &&
                       normalized[2] <= (1u << numBits));

                return (LeftShift3(static_cast<uint32_t>(normalized[2])) << 2) |
                       (LeftShift3(static_cast<uint32_t>(normalized[1])) << 1) |
                       (LeftShift3(static_cast<uint32_t>(normalized[0])));
            });

    thrust::sort_by_key(thrust::device, mortonCodes.begin(), mortonCodes.end(),
                        trias);

    std::cout << "Sorted " << triangles.size() << " Triangles by Morton Code\n";
    BVH *bvh;
    AccelerationNode *bvhNodes;

    checkCudaErrors(cudaMallocManaged(&bvh, sizeof(BVH)));
    checkCudaErrors(cudaMallocManaged(
            &bvhNodes, 2 * triangles.size() * sizeof(AccelerationNode)));

    unsigned short int *bvhConstructionDone;
    checkCudaErrors(
            cudaMallocManaged(&bvhConstructionDone,
                              (triangles.size() - 2) * sizeof(unsigned short int)));

    auto start = std::chrono::high_resolution_clock::now();
    // TODO grid stride loop or tune sizes
    constructBVH<<<(triangles.size() + 255) / 256, 256>>>(
            bvhNodes, trias, triangles.size(), mortonCodes.data().get(),
            bvhConstructionDone);

    checkCudaErrors(cudaDeviceSynchronize());
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::high_resolution_clock::now() - start);

    std::cout << "Built BVH in " << duration.count() << "ms\n";

    *bvh = BVH{bvhNodes};

    scene.bvh = bvh;


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
                logger.template log<false>("<float name=\"fov\" value=\"" + std::to_string(value) + "\"/>");
            } else if(name == "aspectRatio") {
                cameraBuilder.setAspectRatio(value);
                logger.template log<false>("<float name=\"aspectRatio\" value=\"" + std::to_string(value) + "\"/>");
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

    s.camera = cameraBuilder.build();
    sensors.push_back(s);
}
Eigen::Isometry3f SceneBuilder::parseTransform(const pugi::xml_node &node, auto logger) const {
    auto nodeName = std::string(node.name());
    Eigen::Isometry3f tf = Eigen::Isometry3f::Identity();
    auto vecToString = [](const Eigen::Vector3f &vec) {
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
        logger.template log<false>("<matrix value=\"" + (std::ostringstream{} << tf.matrix()).str() + "\"/>");
    } else if(std::string(node.name()) == "lookat") {
        Eigen::Vector3f target = -Eigen::Vector3f::UnitZ(), origin = Eigen::Vector3f::Zero(), up = Eigen::Vector3f::UnitY();
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
Eigen::Vector3f SceneBuilder::parseVector(std::string str) const {
    std::replace(str.begin(), str.end(), ',', ' ');
    std::istringstream stream{str};
    Eigen::Vector3f vec;
    if(!(stream >> vec.x() >> vec.y() >> vec.z())) {
        throw std::runtime_error("Could not parse vector " + str);
    }
    return vec;
}
SceneBuilder &SceneBuilder::getSensors(std::vector<Sensor> &s) {
    s.resize(this->sensors.size());
    for(int i = 0; i < s.size(); i++) {
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


__device__ __host__ constexpr uint32_t LeftShift3(uint32_t x) noexcept {
    if(x == (1 << 10)) --x;
    x = (x | (x << 16)) & 0b00000011000000000000000011111111;
    x = (x | (x << 8)) & 0b00000011000000001111000000001111;
    x = (x | (x << 4)) & 0b00000011000011000011000011000011;
    x = (x | (x << 2)) & 0b00001001001001001001001001001001;
    return x;
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
