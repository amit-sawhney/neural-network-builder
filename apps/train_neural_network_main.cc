#include <iostream>

#include "core/neural_network_model.h"

using neural_network::Matrix;
using neural_network::NeuralNetworkModel;

int main() {

  NeuralNetworkModel model({1, 2, 1}, 0.01f);

  Matrix weights = model.GetModelWeights();

  for (const std::vector<float> &weight_layer : weights) {
    std::cout << weight_layer.size() << std::endl;
  }

  return 0;
}