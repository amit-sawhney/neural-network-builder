#pragma once

#include <vector>
namespace neural_network {

typedef std::vector<std::vector<float>> Matrix;

class NeuralNetworkTrainer {

public:
  NeuralNetworkTrainer();

  NeuralNetworkTrainer(Matrix weights, std::vector<size_t> layer_sizes);

  Matrix ForwardPropagate(const std::vector<float> &neuron_values);

  Matrix BackPropagate(const std::vector<float> &expected_values,
                       const Matrix &neuron_values);

  Matrix weights_;
  std::vector<size_t> layer_sizes_;
};
} // namespace neural_network