#include "visualizer/neural_network_builder.h"

namespace neural_network {
namespace visualizer {

NeuralNetworkBuilderApp::NeuralNetworkBuilderApp() {

  window_width_ = float(GetSystemMetrics(SM_CXFULLSCREEN));
  window_height_ = float(GetSystemMetrics(SM_CYFULLSCREEN));

  ci::app::setWindowSize(int(window_width_), int(window_height_));

  BuildNetworkStructure();
}

void NeuralNetworkBuilderApp::fileDrop(ci::app::FileDropEvent event) {}

void NeuralNetworkBuilderApp::BuildNetworkStructure() {

  if (kLayerSizes.empty()) {
    return;
  }

  float width_interval = window_width_ / (float(kLayerSizes.size()) + 1);

  for (size_t layer = 0; layer < kLayerSizes.size(); ++layer) {

    Layer network_layer;
    size_t layer_size = kLayerSizes[layer];

    if (layer_size == 0) {
      throw std::invalid_argument("Invalid layer size");
    }

    float height_interval = window_height_ / (float(layer_size) + 1);

    for (size_t neuron = 0; neuron < layer_size; ++neuron) {
      Neuron new_neuron =
          BuildDynamicNeuron(layer, neuron, height_interval, width_interval);

      network_layer.emplace_back(new_neuron);
    }

    network_.emplace_back(network_layer);
  }
}

void NeuralNetworkBuilderApp::draw() {

  for (const Layer &layer : network_) {
    for (const Neuron &neuron : layer) {
      neuron.Draw();
    }
  }

  DrawConnections();
}

void NeuralNetworkBuilderApp::DrawConnections() const {

  for (int layer = 0; layer < int(network_.size()) - 1; ++layer) {

    Layer current_layer = network_[layer];
    Layer next_layer = network_[layer + 1];

    for (const Neuron &neuron : current_layer) {
      for (const Neuron &next_neuron : next_layer) {
        ci::gl::drawLine(neuron.GetOutputConnectPoint(),
                         next_neuron.GetInputConnectPoint());
      }
    }
  }
}

float NeuralNetworkBuilderApp::CalculateNeuronRadius(
    float x_pos, float height_interval) const {

  float neuron_margin = 10.0f;
  float remaining_width = window_width_ - x_pos;

  // Determine whether there is less height or width available for the neuron
  float diameter = std::min(height_interval / 2, remaining_width);

  return diameter / 2 - neuron_margin;
}

Neuron NeuralNetworkBuilderApp::BuildDynamicNeuron(size_t current_layer,
                                                   size_t current_neuron,
                                                   float height_interval,
                                                   float width_interval) const {

  glm::vec2 center(width_interval * (float(current_layer + 1)),
                   height_interval * (float(current_neuron + 1)));

  float neuron_radius = CalculateNeuronRadius(center.x, height_interval);
  Neuron new_neuron(center, neuron_radius, ci::Color("white"));

  return new_neuron;
}

} // namespace visualizer

} // namespace neural_network