//
// Created by steinraf on 07.04.25.
//

#pragma once

#include "../common.h"
#include <GL/glew.h>
#include <string>

class Viewport{
public:
    //Default virtual destructor
    virtual ~Viewport() = default;

    void renderFrame(bool synchronize=true);

    [[nodiscard]] virtual std::string getTitle() const = 0;
    [[nodiscard]] virtual bool isDone() const = 0;

    virtual void save() = 0;

private:

    //private render function
    virtual void render() = 0;

};

class OpenGLViewport : public Viewport {
public:
    OpenGLViewport(Scene &scene, unsigned int width, unsigned int height, std::string title);
    ~OpenGLViewport() override;

    [[nodiscard]] std::string getTitle() const override;
    [[nodiscard]] bool isDone() const override;

    void save() override;


private:
    void render() override;

    Scene &scene;
    unsigned width, height;
    std::string title;

    bool is_rendering_done = false;

};