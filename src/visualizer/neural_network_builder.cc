#include "visualizer/neural_network_builder.h"

namespace neural_network {
namespace visualizer {

NeuralNetworkBuilderApp::NeuralNetworkBuilderApp() {

  window_width_ = GetSystemMetrics(SM_CXFULLSCREEN);
  window_height_ = GetSystemMetrics(SM_CYFULLSCREEN);

  ci::app::setWindowSize(window_width_, window_height_);
}

void NeuralNetworkBuilderApp::BuildNetworkStructure() const {

  for (size_t layer_size : kLayerSizes) {

    float neuron_radius = float(window_height_) / layer_size;

    for (size_t neuron = 0; neuron < layer_size; ++neuron) {

    }
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