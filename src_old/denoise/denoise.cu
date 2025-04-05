//
// Created by steinraf on 20/12/22.
//

#include "denoise.h"

//Denoise feature is a bit clumsy to use in current state
//TODO add easier customizability


__global__ void denoiseApplyWeights(Vector3f *output, float *weights, int width, int height){
    int i, j, pixelIndex;
    if(!cudaHelpers::initIndices(i, j, pixelIndex, width, height)) return;
//    if(weights[pixelIndex] == 0.f) {
//#ifndef NDEBUG
//        printf("Output would have been (%f, %f, %f)\n", output[pixelIndex][0], output[pixelIndex][1], output[pixelIndex][2]);
//#endif
//        output[pixelIndex] = Vector3f{0.f, 0.f, 1.f};
//        return;
//    }
//    output[pixelIndex] /= weights[pixelIndex];
}


__global__ void denoise(Vector3f *input, Vector3f *output, FeatureBuffer featureBuffer, float *weights, int width, int height,
                        Vector3f cameraOrigin) {
    int i, j, pixelIndex;
    if(!cudaHelpers::initIndices(i, j, pixelIndex, width, height)) return;

    //        bilateralFilterWiki(input, output, i, j, width, height);
//    bilateralFilterSlides(input, output, featureBuffer, weights, i, j, width, height);
//    return;

    //        auto getNeighbour = [i, j, width, height]__device__ (Vector3f *array, int dx, int dy,
    //                                                             BOUNDARY boundary = BOUNDARY::PERIODIC) {
    //            switch(boundary) {
    //                case BOUNDARY::PERIODIC:
    //                    return array[(j + height + dy) % height * width + (i + width + dx) % width];
    //                case BOUNDARY::REFLECTING:
    //                    //TODO implement?
    //                    assert(false && "Not implemented.");
    ////                    break;
    //                case BOUNDARY::ZERO:
    //                    if(i >= 0 && i < width && j >= 0 && j < height)
    //                        return array[j*width + i];
    //
    //                    return Vector3f{0.f};
    //            }
    //        };

    //        const int m_radius = 2;
    //        const float m_stddev = 0.5f;
    //
    //        const float alpha = -1.0f / (2.0f * m_stddev * m_stddev);
    //        const float constant = std::exp(alpha * m_radius * m_radius);


    //        auto gaussian = [alpha, constant] __device__ (float x){
    //            return CustomRenderer::max(0.0f, std::exp(alpha * x * x) - constant);
    //        };

    //        float integral = 0.f;
    //        Vector3f tmp{0.f};

    //        for(int xNew = -m_radius; xNew <= m_radius; ++xNew) {
    //            for(int yNew = -m_radius + 1; yNew < m_radius; ++yNew){
    //                const float gaussianContrib = gaussian(sqrtf(xNew*xNew + yNew*yNew));
    //                tmp += gaussianContrib * getNeighbour(input, xNew, yNew);
    //                integral += gaussianContrib;
    //            }
    //        }
    //
    //        output[pixelIndex] = tmp/integral;

    //TODO when modifying something here, remember to maybe uncomment weights application

    //        output[pixelIndex] = Vector3f{(featureBuffer[pixelIndex].position-cameraOrigin).norm()/200};
//                    output[pixelIndex] = featureBuffer.normals[pixelIndex].absValues();
                    output[pixelIndex] = featureBuffer.albedos[pixelIndex];
//    output[pixelIndex] = featureBuffer.variances[pixelIndex];
    //        output[pixelIndex] = Color3f(featureBuffer.variances[pixelIndex].norm());
//            constexpr float numSamples = 16384.f;
//            output[pixelIndex] = Vector3f{powf(static_cast<float>(featureBuffer.numSubSamples[pixelIndex])/(numSamples), 2.f)};


    //        if(featureBuffer[pixelIndex].variance.maxCoeff() > 0.1) {
    //            output[pixelIndex] = 0.25 * getNeighbour(input, 0, -1) + 0.25 * getNeighbour(input, 1, 0) + 0.25 * getNeighbour(input, 0, 1) + 0.25 * getNeighbour(input, -1, 0);
    //        }
    //        else {
    //            output[pixelIndex] = input[pixelIndex];
    //        }

    //        output[pixelIndex] = Color3f(featureBuffer[pixelIndex].variance.norm());
}


__device__

__device__ void bilateralFilterSlides(Vector3f *input, Vector3f *output, FeatureBuffer &featureBuffer, float *weights, int i, int j, int width, int height){


    //        constexpr int neighbourDiameter = 21;
    //        constexpr int patchDiameter = 7;

    constexpr int neighbourDiameter = 11;
    constexpr int patchDiameter = 7     ;

    constexpr float k = 0.45f;

    const int pixelI = i, pixelJ = j;


    for(int pixelQI = CustomRenderer::max(0, pixelI - neighbourDiameter /2); pixelQI < CustomRenderer::min(width, pixelI + neighbourDiameter /2 + 1); ++pixelQI){
        for(int pixelQJ = CustomRenderer::max(0, pixelJ - neighbourDiameter /2); pixelQJ < CustomRenderer::min(height, pixelJ + neighbourDiameter /2 + 1); ++pixelQJ) {
            float meanDist = 0.f;
            for(int pI = CustomRenderer::max(0, pixelI - patchDiameter / 2); pI < CustomRenderer::min(width, pixelI + patchDiameter / 2 + 1); ++pI) {
                for(int pJ = CustomRenderer::max(0, pixelJ - patchDiameter / 2); pJ < CustomRenderer::min(height, pixelJ + patchDiameter / 2 + 1); ++pJ) {
                    const int pIndex = pJ * width + pI;
                    const Vector3f pVarianceMean = featureBuffer.variances[pIndex]/static_cast<float>(featureBuffer.numSubSamples[pIndex]);

                    for(int qI = CustomRenderer::max(0, pixelQI - patchDiameter / 2); qI < CustomRenderer::min(width, pixelQI + patchDiameter / 2 + 1); ++qI) {
                        for(int qJ = CustomRenderer::max(0, pixelQJ - patchDiameter / 2); qJ < CustomRenderer::min(height, pixelQJ + patchDiameter / 2 + 1); ++qJ) {
                            const int qIndex = qJ * width + qI;
                            const Vector3f qVarianceMean = featureBuffer.variances[qIndex]/static_cast<float>(featureBuffer.numSubSamples[qIndex]);

                            for(int col = 0; col < 3; ++col) {
                                meanDist += (powf(input[pIndex][col] - input[qIndex][col], 2) - (pVarianceMean[col] + CustomRenderer::min(qVarianceMean[col], pVarianceMean[col]))) / (EPSILON + k * k * (pVarianceMean[col] + qVarianceMean[col]));
                            }

//                            Vector3f minVec{
//                                    CustomRenderer::min(qVarianceMean[0], pVarianceMean[0]),
//                                    CustomRenderer::min(qVarianceMean[1], pVarianceMean[1]),
//                                    CustomRenderer::min(qVarianceMean[2], pVarianceMean[2]),
//                            };
//                            meanDist += ((input[pIndex] - input[qIndex])*(input[pIndex] - input[qIndex]) - (pVarianceMean + minVec) / (Vector3f{EPSILON} + k * k * (pVarianceMean + qVarianceMean))).norm();

                            //meanDist .= ((input[pI] - input[qI])^2 - (var[pI] + min(var[pI], var[qI])
                            //            --------------------------------------------------------
                            //              (EPSILON + k^2 * (var[pI] + var[qI]))
                        }
                    }
                }
            }

//            printf("Mean distance is %f\n", meanDist/(3 * patchDiameter * patchDiameter));
            float w = expf(-CustomRenderer::max(0.f, meanDist/(3 * patchDiameter * patchDiameter)));

            for(int pI = CustomRenderer::max(0, pixelI - patchDiameter / 2); pI < CustomRenderer::min(width, pixelI + patchDiameter / 2 + 1); ++pI) {
                for(int pJ = CustomRenderer::max(0, pixelJ - patchDiameter / 2); pJ < CustomRenderer::min(height, pixelJ + patchDiameter / 2 + 1); ++pJ) {

                    const int pIndex = pJ * width + pI;

                    for(int qI = CustomRenderer::max(0, pixelQI - patchDiameter / 2); qI < CustomRenderer::min(width, pixelQI + patchDiameter / 2 + 1); ++qI) {
                        for(int qJ = CustomRenderer::max(0, pixelQJ - patchDiameter / 2); qJ < CustomRenderer::min(height, pixelQJ + patchDiameter / 2 + 1); ++qJ) {

                            const int qIndex = qJ * width + qI;

                            atomicAdd(weights + pIndex, w);
                            Vector3f::atomicCudaAdd(output + pIndex, w * input[qIndex]);
                        }
                    }
                }
            }
        }
    }
}