//
// Created by steinraf on 24/10/22.
//

#include "meshLoader.h"
#include "vector.cuh"
#include <thrust/transform_scan.h>

HostMeshInfo loadMesh(const std::filesystem::path &filePath, const Matrix4f &transform) noexcept(false) {

    std::cout << "Reading mesh " + filePath.filename().string() + " ...\n";
    std::ifstream file(filePath, std::ios::in);
    if(!file.is_open())
        throw std::runtime_error("Error, file " + filePath.string() + " could not be opened in the MeshLoader.");

    std::string lineString;

    thrust::host_vector<Vector3f> vertices;
    thrust::host_vector<Vector2f> textures;
    thrust::host_vector<Vector3f> normals;


    thrust::host_vector<int> vertexIndices1;
    thrust::host_vector<int> vertexIndices2;
    thrust::host_vector<int> vertexIndices3;

    thrust::host_vector<int> textureIndices1;
    thrust::host_vector<int> textureIndices2;
    thrust::host_vector<int> textureIndices3;

    thrust::host_vector<int> normalIndices1;
    thrust::host_vector<int> normalIndices2;
    thrust::host_vector<int> normalIndices3;


    while(std::getline(file, lineString)) {
        std::istringstream line{lineString};
        std::string start;

        line >> start;

        if(start == "v") {
            float x, y, z;
            line >> x >> y >> z;
            vertices.push_back(Vector3f{x, y, z}.applyTransform(transform));
            //            std::cout << filePath << ", found coord x y z " << x << ' ' << y << ' ' << z << '\n';

        } else if(start == "vt") {
            float u, v, w;
            line >> u >> v >> w;
            textures.push_back({u, v});
            //            std::cout << filePath << ", found texture u v w " << u << ' ' << v << ' ' << w << '\n';
        } else if(start == "vn") {
            float x, y, z;
            line >> x >> y >> z;
            normals.push_back(Vector3f{x, y, z}.applyTransform(transform, true));

        } else if(start == "f") {
            std::string e1, e2, e3, e4;
            line >> e1 >> e2 >> e3 >> e4;

            std::istringstream s1(e1), s2(e2), s3(e3);

            int v1, v2, v3,
                    t1, t2, t3,
                    n1, n2, n3;

            char delim;

            s1 >> v1 >> delim >> t1 >> delim >> n1;
            s2 >> v2 >> delim >> t2 >> delim >> n2;
            s3 >> v3 >> delim >> t3 >> delim >> n3;

            if(v1 < 0) v1 += vertices.size();
            if(v2 < 0) v2 += vertices.size();
            if(v3 < 0) v3 += vertices.size();

            if(t1 < 0) t1 += textures.size();
            if(t2 < 0) t2 += textures.size();
            if(t3 < 0) t3 += textures.size();

            if(n1 < 0) n1 += normals.size();
            if(n2 < 0) n2 += normals.size();
            if(n3 < 0) n3 += normals.size();

//            std::cout << "Adding face vertices " << v1 << ' ' << v2 << ' ' << v3 << '\n';

            vertexIndices1.push_back(v1 - 1);
            vertexIndices2.push_back(v2 - 1);
            vertexIndices3.push_back(v3 - 1);

            textureIndices1.push_back(t1 - 1);
            textureIndices2.push_back(t2 - 1);
            textureIndices3.push_back(t3 - 1);

            normalIndices1.push_back(n1 - 1);
            normalIndices2.push_back(n2 - 1);
            normalIndices3.push_back(n3 - 1);

            if(!e4.empty()) {
                int v4, t4, n4;
                std::istringstream s4(e4);
                s4 >> v4 >> delim >> t4 >> delim >> n4;
                assert(delim == '/');

                if(v4 < 0) v4 += vertices.size();
                if(t4 < 0) t4 += textures.size();
                if(n4 < 0) n4 += normals.size();

//                std::cout << "Adding face vertices " << v4 << ' ' << v1 << ' ' << v3 << '\n';


                vertexIndices3.push_back(v4 - 1);
                vertexIndices1.push_back(v1 - 1);
                vertexIndices2.push_back(v3 - 1);


                textureIndices3.push_back(t4 - 1);
                textureIndices1.push_back(t1 - 1);
                textureIndices2.push_back(t3 - 1);


                normalIndices3.push_back(n4 - 1);
                normalIndices1.push_back(n1 - 1);
                normalIndices2.push_back(n3 - 1);
            }
        }
    }


    if(normals.size() == 0) {
        normals.resize(vertexIndices1.size());
        std::cout << "\tNo normals found. Interpolating...\n";
        for(size_t i = 0; i < vertexIndices1.size(); ++i) {
            const Vector3f p0 = vertices[vertexIndices1[i]],
                           p1 = vertices[vertexIndices2[i]],
                           p2 = vertices[vertexIndices3[i]];
            normals[i] = ((p1 - p0).cross(p2 - p0)).normalized();
        }
    }

    if(textures.size() == 0) {
        textures.resize(textureIndices1.size());
        std::cout << "\tNo textures found. Interpolating...\n";
        for(size_t i = 0; i < textureIndices1.size(); ++i) {
            textures[i] = {0.f, 0.f};
        }
    }

    return {
            vertices,
            textures,
            normals,
            {vertexIndices1, vertexIndices2, vertexIndices3},
            {textureIndices1, textureIndices2, textureIndices3},
            {normalIndices1, normalIndices2, normalIndices3}};
}


DeviceMeshInfo meshToGPU(const HostMeshInfo &mesh) noexcept {
    const auto numTriangles = mesh.normalsIndices.first.size();

    std::vector<Triangle> hostTriangles(numTriangles);

    const auto &[vertices,
                 textures,
                 normals,
                 vertexIndexList,
                 textureIndexList,
                 normalIndexList] = mesh;


#pragma omp parallel for
    for(size_t i = 0; i < numTriangles; ++i) {
        hostTriangles[i] = {
                vertices[vertexIndexList.first[i]],
                vertices[vertexIndexList.second[i]],
                vertices[vertexIndexList.third[i]],
                textures[textureIndexList.first[i]],
                textures[textureIndexList.second[i]],
                textures[textureIndexList.third[i]],
                normals[normalIndexList.first[i]],
                normals[normalIndexList.second[i]],
                normals[normalIndexList.third[i]],
        };
    }

    thrust::device_vector<Triangle> deviceTriangles(hostTriangles);


    TriaToAABB triangleToAABB;
    AABB aabb{};
    AABBAdder aabbAddition;

    TriaToArea triangleToArea;


    AABB maxBoundingBox = thrust::transform_reduce(deviceTriangles.begin(), deviceTriangles.end(),
                                                   triangleToAABB, aabb, aabbAddition);

    float totalTriaArea = thrust::transform_reduce(deviceTriangles.begin(), deviceTriangles.end(),
                                                   triangleToArea, 0.f, thrust::plus<float>());

    assert(isfinite(totalTriaArea));

    TriangleToCDF triangleToCdf(totalTriaArea);
    thrust::device_vector<float> areaCDF(numTriangles);

    thrust::transform_inclusive_scan(deviceTriangles.begin(), deviceTriangles.end(),
                                     areaCDF.begin(), triangleToCdf, thrust::plus<float>());

    printf("\tTotal area of all triangles is %f\n", totalTriaArea);


    TriangleToMortonCode triangleToMortonCode(maxBoundingBox);

    thrust::device_vector<uint32_t> mortonCodes(numTriangles);

    thrust::transform(deviceTriangles.begin(), deviceTriangles.end(), mortonCodes.begin(),
                      triangleToMortonCode);

    thrust::sort_by_key(mortonCodes.begin(), mortonCodes.end(), deviceTriangles.begin());
    //TODO maybe use radix sort

    return {deviceTriangles, mortonCodes, areaCDF, totalTriaArea};
}


__host__ BLAS *getMeshFromFile(const std::string &filename, thrust::device_vector<Triangle> &deviceTrias,
                               thrust::device_vector<float> &areaCDF, float &totalArea,
                               const Matrix4f &transform,
                               BSDF bsdf, Texture normalMap, AreaLight *deviceEmitter, GalaxyMedium *medium) noexcept(false) {
    clock_t startGeometryBVH = clock();


    auto mesh = loadMesh(filename, transform);
    auto [deviceTriangles, deviceMortonCodes, deviceCDF, area] = meshToGPU(mesh).toTuple();
    totalArea = area;
    Triangle *deviceTriaPtr = deviceTriangles.data().get();

    const size_t numTriangles = mesh.vertexIndices.first.size();

    std::cout << "\tLoading Geometry took "
              << ((double) (clock() - startGeometryBVH)) / CLOCKS_PER_SEC
              << " seconds.\n";

    std::cout << "\tManaging " << numTriangles << " triangles.\n";

    clock_t bvhConstructStart = clock();

    BLAS *bvh;

    AccelerationNode *bvhTotalNodes;
    checkCudaErrors(cudaMalloc((void **) &bvhTotalNodes,
                               sizeof(AccelerationNode) * (2 * numTriangles - 1)));//n-1 internal, n leaf
    checkCudaErrors(cudaMalloc((void **) &bvh, sizeof(BLAS)));


    cudaHelpers::constructBVH<<<(numTriangles + 1024 - 1) / 1024, 1024>>>
            (bvhTotalNodes, deviceTriaPtr, deviceMortonCodes.data().get(), numTriangles);


    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    std::cout << "\tBLAS construction took "
              << ((double) (clock() - bvhConstructStart)) / CLOCKS_PER_SEC
              << " seconds.\n";

    clock_t bvhBoundingBox = clock();

    cudaHelpers::computeBVHBoundingBoxes<<<1, 1>>>(bvhTotalNodes);

    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    std::cout << "\tBLAS boundingBox Compute took "
              << ((double) (clock() - bvhBoundingBox)) / CLOCKS_PER_SEC
              << " seconds.\n";

    clock_t initBVHTime = clock();

    cudaHelpers::initBVH<<<1, 1>>>(bvh, bvhTotalNodes, totalArea, deviceCDF.data().get(), numTriangles, deviceEmitter, bsdf, normalMap, medium);

    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    std::cout << "\tBLAS init took "
              << ((double) (clock() - initBVHTime)) / CLOCKS_PER_SEC
              << " seconds.\n";

    std::cout << "\tLoading Geometry and BLAS construction took "
              << ((double) (clock() - startGeometryBVH)) / CLOCKS_PER_SEC
              << " seconds.\n";

    deviceTrias = std::move(deviceTriangles);
    areaCDF = std::move(deviceCDF);

    return bvh;
}