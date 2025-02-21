//
// Created by steinraf on 17.02.25.
//

#pragma once


// Forward declarations

struct AABB;
class BVH;
class Camera;
struct FeatureBuffer;
class Frame;
struct Intersection;
class Ray;
class Sampler;
class Scene;
class SceneBuilder;
class Triangle;


void checkCudaErrors(cudaError result);
