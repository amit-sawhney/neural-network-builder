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
  const std::vector<size_t> kLayerSizes{2, 2, 1};

  void BuildNetworkStructure() const;

  float CalculateSpaceBetweenLayers() const;

  float CalculateSpaceBetweenNeurons() const;

  float CalculateNeuronSize() const;

  void DrawConnections() const;

  int window_height_;
  int window_width_;
  Network network_;
};
} // namespace visualizer

} // namespace neural_network
