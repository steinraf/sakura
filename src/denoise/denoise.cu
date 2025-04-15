//
// Created by steinraf on 20/12/22.
//

#include "denoise.h"
#include <optional>

//Denoise feature is a bit clumsy to use in current state
//TODO add easier customizability

#define DENOISER_EPSILON std::numeric_limits<float>::epsilon()

GPU_ONLY void bilateralFilterSlides(FeatureBuffer *featureBuffer, Vec3f *output, float *weights, int i, int j, int width, int height){


//    constexpr int neighbourDiameter = 21;
//    constexpr int patchDiameter = 7;

    constexpr int neighbourDiameter = 5;
    constexpr int patchDiameter = 3;

    constexpr float k = 0.45f;

    const int pixelI = i, pixelJ = j;


    for(int pixelQI = max(0, pixelI - neighbourDiameter /2); pixelQI < min(width, pixelI + neighbourDiameter /2 + 1); ++pixelQI){
        for(int pixelQJ = max(0, pixelJ - neighbourDiameter /2); pixelQJ < min(height, pixelJ + neighbourDiameter /2 + 1); ++pixelQJ) {
            float meanDist = 0.f;
            for(int pI = max(0, pixelI - patchDiameter / 2); pI < min(width, pixelI + patchDiameter / 2 + 1); ++pI) {
                for(int pJ = max(0, pixelJ - patchDiameter / 2); pJ < min(height, pixelJ + patchDiameter / 2 + 1); ++pJ) {
                    const int pIndex = pJ * width + pI;


                    for(int qI = max(0, pixelQI - patchDiameter / 2); qI < min(width, pixelQI + patchDiameter / 2 + 1); ++qI) {
                        for(int qJ = max(0, pixelQJ - patchDiameter / 2); qJ < min(height, pixelQJ + patchDiameter / 2 + 1); ++qJ) {
                            const int qIndex = qJ * width + qI;

                            auto computeFeature = [&qIndex, &pIndex](Statistic<Vec3f> *stat) -> float {
                                const Vector3f qMean = stat[qIndex].getMean();
                                const Vector3f pMean = stat[pIndex].getMean();
                                const Vector3f qVariance = stat[qIndex].getSampleMeanVariance();
                                const Vector3f pVariance = stat[pIndex].getSampleMeanVariance();

                                float weight_p = 0.f;
                                for(int col = 0; col < 3; ++col) {
                                    weight_p += (powf(pMean[col] - qMean[col], 2) - (pVariance[col] + min(qVariance[col], pVariance[col])))
                                                / (EPSILON + k * k * (pVariance[col] + qVariance[col]));
                                }
                                return weight_p;
                                //meanDist .= ((input[pI] - input[qI])^2 - (var[pI] + min(var[pI], var[qI])
                                //            --------------------------------------------------------
                                //              (EPSILON + k^2 * (var[pI] + var[qI]))
                            };

                            meanDist += computeFeature(featureBuffer->color)
//                                        * computeFeature(featureBuffer->position)
//                                        * computeFeature(featureBuffer->normal)
                                    ;

                        }
                    }
                }
            }

            float w = expf(-max(0.f, meanDist/(3 * patchDiameter * patchDiameter)));

            if(w == 0) continue;

            for(int pI = max(0, pixelI - patchDiameter / 2); pI < min(width, pixelI + patchDiameter / 2 + 1); ++pI) {
                for(int pJ = max(0, pixelJ - patchDiameter / 2); pJ < min(height, pixelJ + patchDiameter / 2 + 1); ++pJ) {

                    const int pIndex = pJ * width + pI;

                    for(int qI = max(0, pixelQI - patchDiameter / 2); qI < min(width, pixelQI + patchDiameter / 2 + 1); ++qI) {
                        for(int qJ = max(0, pixelQJ - patchDiameter / 2); qJ < min(height, pixelQJ + patchDiameter / 2 + 1); ++qJ) {

                            const int qIndex = qJ * width + qI;
                            auto add = []__device__(Vector3f *address, const Vector3f &vec){
                                Vector3f &v = *address;
                                atomicAdd(&(v[0]), vec[0]);
                                atomicAdd(&(v[1]), vec[1]);
                                atomicAdd(&(v[2]), vec[2]);
                            };

                            atomicAdd(weights + pIndex, w);
                            add(output + pIndex, w * featureBuffer->color[qIndex].getMean());
                        }
                    }
                }
            }
        }
    }
}

GPU_ONLY void removeFireflies(FeatureBuffer *bufferIn, Vec3f *output, float *weights, unsigned int x, unsigned int y, unsigned int width, unsigned int height) {

    Color3f localMean = Color3f::Zero();
    Vec3f localVariance = Vec3f::Zero();
    float MAX_DIFF = 1.0f;
    float FIREFLY_SCALE = 2.0f;
    int numSamples = 0;
    constexpr int MAX_RADIUS = 1;
    for(int dx = -MAX_RADIUS; dx <= MAX_RADIUS; ++dx) {
        for(int dy = -MAX_RADIUS; dy <= MAX_RADIUS; ++dy) {
            int nx = x + dx, ny = y + dy;
            if(nx < 0 || nx >= width || ny < 0 || ny >= height || (dy == 0 && dx == 0)) {
                continue;
            }
            const Vec3f var = bufferIn->color[nx + ny * width].getSampleVariance();
            localVariance += var;
            localMean += bufferIn->color[nx + ny * width].getMean() ;

            ++numSamples;
        }
    }

    localVariance /= numSamples;
    localMean /= numSamples;

    Vec3f color = bufferIn->color[x + y * width].getMean();
    Vec3f var = bufferIn->color[x + y * width].getSampleVariance();
    auto nSamples = bufferIn->color[x + y * width].getNumElements();

    if(FIREFLY_SCALE * localVariance.norm() < var.norm()){
        float weight = 1.0f - tanh(var.norm() - FIREFLY_SCALE * localVariance.norm());
        output[x + y * width] = weight * color + (1.0f - weight) * localMean;
        weights[x + y * width] = 1.0f;

    }else{
        weights[x + y * width] = 0.0f;
    }

//    float diff = (color - localMean).norm();
//
//    if(diff > MAX_DIFF) {
//        bufferIn->color[x + y * width].manipulateMean(localMean);
//    }

}

GPU_ONLY void denoiseGaussian(const FeatureBuffer *bufferIn, Vec3f *output, float *weights, unsigned int x, unsigned int y, unsigned int width, unsigned int height) {
    size_t pixelIndex = y * width + x;

    // Simply apply gaussian blur
    int MAX_RADIUS = 3;
    constexpr float color_k = 0.7f, spatial_k = 0.005f, var_k = 1.0f;

    Vec3f color = Vec3f::Zero();
    float weight = 0.0f;

    Vec3f pos = bufferIn->position[pixelIndex].getMean();
    Vec3f var = bufferIn->color[pixelIndex].getSampleVariance();
    Vec3f normal = bufferIn->normal[pixelIndex].getMean();
    size_t numSamples = bufferIn->color[pixelIndex].getNumElements();

    MAX_RADIUS = min(MAX_RADIUS, int(numSamples));

    for(int dx = -MAX_RADIUS; dx <= MAX_RADIUS; ++dx) {
        for(int dy = -MAX_RADIUS; dy < MAX_RADIUS; ++dy) {
            int nx = x + dx, ny = y + dy;
            if(nx < 0 || nx >= width || ny < 0 || ny >= height) {
                continue;
            }

            Vec3f currentColor = bufferIn->color[nx + ny * width].getMean();
            Vec3f currentVar = bufferIn->color[nx + ny * width].getSampleVariance();
            Vec3f currentNormal = bufferIn->normal[nx + ny * width].getMean();

            const float distanceSq = Vec2f{dx * 1.0f, dy * 1.0f}.squaredNorm();
            const float spatialDistSq = (bufferIn->position[nx + ny * width].getMean() - pos).squaredNorm();
            const float normalDot = std::abs(normal.dot(currentNormal));

            const float varRatio = [&]() -> float {
                if(var.minCoeff() < DENOISER_EPSILON) {
                    return currentVar.maxCoeff() / DENOISER_EPSILON;
                } else {
                    return (currentVar / var).maxCoeff();
                }
            }();

            const float gaussianScreenSpace = std::exp(-distanceSq / (color_k * color_k));
            const float gaussianSpatial = std::exp(-spatialDistSq / (spatial_k * spatial_k));
            const float gaussianVar = std::exp(-varRatio / (var_k * var_k));

            float w = [&](){
                if(numSamples < 4) {
                    return gaussianScreenSpace;
                }else{
                    return gaussianSpatial * normalDot;
                }
            }();

            color += currentColor * w;
            weight += w;
        }
    }

    if(weight < DENOISER_EPSILON) weight = DENOISER_EPSILON;

    Vec3f localAverageColor = color / weight;

    output[pixelIndex] = localAverageColor;
    weights[pixelIndex] = 0.0f;
}

template <typename T>
struct TemporaryStatisticUpdate{
public:
    CPU_GPU_CONSTEXPR TemporaryStatisticUpdate() noexcept
        : stat(std::nullopt) {

    }
    CPU_GPU_CONSTEXPR explicit TemporaryStatisticUpdate(Statistic<T> &stat) : stat({&stat, stat}) {

    }

    CPU_GPU_CONSTEXPR ~TemporaryStatisticUpdate() {
        if(stat.has_value()){
            *(stat->stat) = stat->initialValue;
        }
    }

    CPU_GPU_CONSTEXPR TemporaryStatisticUpdate(const TemporaryStatisticUpdate &) = delete;
    CPU_GPU_CONSTEXPR TemporaryStatisticUpdate &operator=(const TemporaryStatisticUpdate &) = delete;

    CPU_GPU_CONSTEXPR TemporaryStatisticUpdate(TemporaryStatisticUpdate&& other) noexcept
        : stat(std::move(other.stat)) {
        other.stat = std::nullopt;
    }

    CPU_GPU_CONSTEXPR TemporaryStatisticUpdate& operator=(TemporaryStatisticUpdate&& other) noexcept {
        if (this != &other) {
            stat = std::move(other.stat);
            other.stat = std::nullopt;
        }
        return *this;
    }

    CPU_GPU_CONSTEXPR void manipulateMean(const T &value) {
        if(stat.has_value()){
            stat->stat->manipulateMean(value);
        }
    }



private:
    struct Binding {
        Statistic<T> *stat;
        Statistic<T> initialValue;
    };
    std::optional<Binding> stat;
};


__global__ void denoiser(FeatureBuffer *featureBuffer, Vec3f *output, float *weights, int width, int height) {
    int i, j, pixelIndex;
    if(!cudaHelpers::initIndices(i, j, pixelIndex, width, height)) return;

//    removeFireflies(featureBuffer, output, weights, i, j, width, height);
//    TemporaryStatisticUpdate<Vec3f> statUpdate;
//
//    if(weights[pixelIndex] != 0.0){
//        statUpdate = TemporaryStatisticUpdate<Vec3f>(featureBuffer->color[pixelIndex]);
//        statUpdate.manipulateMean(output[pixelIndex]);
//        weights[pixelIndex] = 0.0f;
//    }
//    __syncthreads();
//    denoiseGaussian(featureBuffer, output, weights, i, j, width, height);
    bilateralFilterSlides(featureBuffer, output, weights, i, j, width, height);

//    output[pixelIndex] = featureBuffer->color[pixelIndex].getMean();


}

__global__ void denoiseApplyWeights(FeatureBuffer *featureBuffer, Vec3f *output, float *weights, int width, int height) {
    int i, j, pixelIndex;
    if(!cudaHelpers::initIndices(i, j, pixelIndex, width, height)) return;


    if(weights[pixelIndex] < DENOISER_EPSILON) {
#ifndef NDEBUG
        output[pixelIndex] = Vector3f{0.f, 0.f, 1.f};
#else
        output[pixelIndex] = featureBuffer->color[pixelIndex].getMean();
#endif
        return;
    }
    output[pixelIndex] /= weights[pixelIndex];
}