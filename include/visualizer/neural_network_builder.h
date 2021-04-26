#pragma once

#include <Windows.h>

#include "cinder/app/App.h"
#include "cinder/app/RendererGl.h"
#include "cinder/gl/gl.h"
#include "core/model.h"
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
  const std::vector<size_t> kLayerSizes{2, 1};

  void BuildNetworkStructure();

  float CalculateNeuronRadius(float x_pos, float height_interval) const;

  Neuron BuildDynamicNeuron(size_t current_layer, size_t current_neuron,
                            float height_interval, float width_interval) const;

  void DrawConnections() const;

  void TrainModel(std::ifstream *training_data);

  void Predict(std::ifstream *training_data);

  void UpdateVisualNeuralNetworkValues(const neural_network::Layer &output_values);

  float window_height_;
  float window_width_;
  Network network_;
  ci::Color neuron_color_;
  Model network_model_;
  neural_network::Layer output_values_;
  float learning_rate_;
};
} // namespace visualizer

} // namespace neural_network
