#include "visualizer/neural_network_builder.h"

using neural_network::visualizer::NeuralNetworkBuilderApp;

void prepareSettings(NeuralNetworkBuilderApp::Settings *settings) {
  settings->setResizable(false);
}

// This line is a macro that expands into an "int main()" function.
CINDER_APP(NeuralNetworkBuilderApp, ci::app::RendererGl, prepareSettings);