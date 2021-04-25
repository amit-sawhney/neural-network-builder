#pragma once

#include "cinder/app/App.h"
#include "cinder/app/RendererGl.h"
#include "cinder/gl/gl.h"

namespace neural_network {

namespace visualizer {

class NeuralNetworkBuilderApp : public ci::app::App {
public:
  NeuralNetworkBuilderApp();

  void draw() override;

  const float kWindowSize = 1075.0f;

private:
};
} // namespace visualizer
} // namespace neural_network
