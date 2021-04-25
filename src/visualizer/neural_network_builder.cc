#include "visualizer/neural_network_builder.h"

namespace neural_network {
namespace visualizer {

NeuralNetworkBuilderApp::NeuralNetworkBuilderApp() {

  window_width_ = GetSystemMetrics(SM_CXFULLSCREEN);
  window_height_ = GetSystemMetrics(SM_CYFULLSCREEN);

  ci::app::setWindowSize(window_width_, window_height_);
}

void NeuralNetworkBuilderApp::draw() {

  Neuron neuron(glm::vec2(500, 500), 250, ci::Color("white"));
  neuron.Draw();
}

} // namespace visualizer

} // namespace neural_network