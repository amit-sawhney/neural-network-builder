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

  void fileDrop(ci::app::FileDropEvent event) override;

private:
  const std::vector<size_t> kLayerSizes{20, 10, 5, 1};

  void BuildNetworkStructure();

  float CalculateNeuronRadius(float x_pos, float height_interval) const;

  Neuron BuildDynamicNeuron(size_t current_layer, size_t current_neuron,
                            float height_interval, float width_interval) const;

  void DrawConnections() const;

  float window_height_;
  float window_width_;
  Network network_;
  ci::Color neuron_color_;
};
} // namespace visualizer

} // namespace neural_network
