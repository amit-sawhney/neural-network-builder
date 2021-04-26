#include "visualizer/neural_network_builder.h"

namespace neural_network {
namespace visualizer {

NeuralNetworkBuilderApp::NeuralNetworkBuilderApp() {

  window_width_ = float(GetSystemMetrics(SM_CXFULLSCREEN));
  window_height_ = float(GetSystemMetrics(SM_CYFULLSCREEN));

  ci::app::setWindowSize(int(window_width_), int(window_height_));

  BuildNetworkStructure();
}

void NeuralNetworkBuilderApp::BuildNetworkStructure() {

  if (kLayerSizes.empty()) {
    return;
  }

  glm::vec2 screen_center(window_width_ / 2, window_height_ / 2);
  float width_start = window_width_ / (float(kLayerSizes.size()) + 1);

  for (size_t layer = 0; layer < kLayerSizes.size(); ++layer) {

    Layer network_layer;
    size_t layer_size = kLayerSizes[layer];

    if (layer_size == 0) {
      throw std::invalid_argument("Invalid layer size");
    }

    float height_interval = window_height_ / (float(layer_size) + 1);

    for (size_t neuron = 0; neuron < layer_size; ++neuron) {
      glm::vec2 center(width_start * (float(layer + 1)),
                       height_interval * (float(neuron + 1)));

      float neuron_radius =
          std::min(height_interval / 2, window_width_ - center.x) / 2 - 10;
      Neuron new_neuron(center, neuron_radius, ci::Color("white"));
      network_layer.emplace_back(new_neuron);
    }

    network_.emplace_back(network_layer);
  }
}

void NeuralNetworkBuilderApp::draw() {

  for (const auto &neurons : network_) {
    for (const Neuron &neuron : neurons) {
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

float NeuralNetworkBuilderApp::CalculateNeuronSize() const { return 0.0f; }

float NeuralNetworkBuilderApp::CalculateSpaceBetweenLayers() const {
  return 0.0f;
}

float NeuralNetworkBuilderApp::CalculateSpaceBetweenNeurons() const {
  return 0.0f;
}

} // namespace visualizer

} // namespace neural_network