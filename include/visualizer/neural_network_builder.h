#pragma once

#include <Windows.h>

#include "cinder/app/App.h"
#include "cinder/app/RendererGl.h"
#include "cinder/gl/gl.h"
#include "neuron.h"

namespace neural_network {

namespace visualizer {

class NeuralNetworkBuilderApp : public ci::app::App {

public:
  NeuralNetworkBuilderApp();

  void draw() override;

private:
  int window_height_;
  int window_width_;
  std::vector<size_t> layer_sizes_{2, 2, 1};
};
} // namespace visualizer

} // namespace neural_network
