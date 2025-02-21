//
// Created by steinraf on 04.02.25.
//


#include <thrust/device_vector.h>
#include <thrust/sort.h>

#include "../acceleration/aabb.cuh"
#include "../acceleration/bvh.cuh"
#include "../camera/camera.cuh"

#include "../integrator/integrators.cuh"
#include "scene.cuh"

#include "pugixml.hpp"


void Scene::render(cudaSurfaceObject_t surface, FeatureBuffer *buffer, Camera &camera, curandState *rngStates, const ImVec2 &windowSize, int spp) const {


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

    std::vector<Triangle> triangles{};

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
                faceVertices[i][2] *= -1;//TODO remove
                std::string textureIndex;
                std::getline(vertexStream, textureIndex, '/');
                //                uvTextures[i] = uvs[std::stoi(textureIndex) -
                //                1];
                std::string normalIndex;
                std::getline(vertexStream, normalIndex, '/');
                faceNormals[i] = normals[std::stoi(normalIndex) - 1];
            }


            triangles.emplace_back(
                    tf * faceVertices[0], tf * faceVertices[1],
                    tf * faceVertices[2], tf.linear() * faceNormals[0],
                    tf.linear() * faceNormals[1], tf.linear() * faceNormals[2]);
        }
    }
    for(const auto &t: triangles) {
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

    auto [doc, root] = loadXML(filename);

    const auto &rootLogger = sceneLogger.getNewSection("scene");

#define CREATE_PARSER(name) {#name, [&](const pugi::xml_node &node, const auto &logger) { parse_##name(node, logger); }}

    static const std::unordered_map<std::string, std::function<void(const pugi::xml_node &, const ScopedLogger &)>>
            parsers{
                    CREATE_PARSER(shape),
            };

    for(const auto &node: root.children()) {
        if(node.type() == pugi::node_comment ||
           node.type() == pugi::node_declaration)
            continue;
        if(node.type() != pugi::node_element)
            throw std::runtime_error("Unknown XML Node encountered.");


        std::string name = node.name();

        const auto &logger = sceneLogger.getNewSection(name);


        if(auto it = parsers.find(name); it != parsers.end()) {
            it->second(node, logger);
        } else {
            logger.log<true>("Warning: Ignoring XML Node \"" + name + "\"");
        }
    }
    return *this;
}

std::pair<pugi::xml_document, pugi::xml_node> SceneBuilder::loadXML(const std::string &filename) const noexcept(false) {
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


void SceneBuilder::parse_shape(const pugi::xml_node &shape, const auto &logger) {

    auto attribute = lookupName(shape.attribute("type").value());
    if(attribute == "obj") {


        auto filenameNode = shape.find_child([](const pugi::xml_node &attrib) {
            return std::string(attrib.name()) == "string";
        });
        auto filename = lookupName(filenameNode.attribute("value").value());

        if(lookupName(filenameNode.attribute("name").value()) != "filename") {
            throw std::runtime_error("String attribute should have name \"filename\", not " + filename);
        }

        addObj(filename);

        logger.template log<false>("FOUND OBJ " + std::string(filename));
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

    *bvh = BVH{bvhNodes, triangles.size()};

    scene.bvh = bvh;


    return scene;
}
std::string SceneBuilder::lookupName(const std::string &name) const {
    if(name.empty()) return name;
    if(name[0] != '$') return name;
    try {
        return nameMap.at(name);
    } catch(const std::out_of_range &e) {
        throw std::runtime_error("Unknown XML name alias " + name);
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
ScopedLogger::ScopedLogger(SceneLogger &formatter, std::string tagName, std::string attribute) : formatter(formatter), tagName(std::move(tagName)) {
    log<false>('<' + this->tagName + (attribute == "" ? "" : " ") + attribute + '>');
    formatter.indent();
}
ScopedLogger ScopedLogger::getNewSection(std::string tagName, std::string attribute) {
    return formatter.getNewSection(tagName, attribute);
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

ScopedLogger SceneLogger::getNewSection(std::string tagName, std::string attribute) {
    return ScopedLogger{*this, tagName, attribute};
}
