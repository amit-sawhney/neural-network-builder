#pragma once

#include <Windows.h>

#include "cinder/app/App.h"
#include "cinder/app/RendererGl.h"
#include "cinder/gl/gl.h"
#include "neuron.h"

namespace neural_network {

namespace visualizer {

typedef std::vector<Neuron> Layer;
typedef std::vector<Layer> Network;

class NeuralNetworkBuilderApp : public ci::app::App {

public:
  NeuralNetworkBuilderApp();

  void draw() override;

private:
  const std::vector<size_t> kLayerSizes{5, 4, 3, 2, 3, 4, 5};

  void BuildNetworkStructure();

  float CalculateSpaceBetweenLayers() const;

  float CalculateSpaceBetweenNeurons() const;

  float CalculateNeuronSize() const;

  void DrawConnections() const;

  float window_height_;
  float window_width_;
  Network network_;
};
} // namespace visualizer

} // namespace neural_network
