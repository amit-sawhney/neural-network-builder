#include "visualizer/neural_network_builder.h"

using neural_network::visualizer::NeuralNetworkBuilderApp;

void prepareSettings(NeuralNetworkBuilderApp::Settings *settings) {
  settings->setResizable(false);
}

CINDER_APP(NeuralNetworkBuilderApp, ci::app::RendererGl, prepareSettings);
