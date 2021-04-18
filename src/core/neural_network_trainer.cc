#include "core/neural_network_trainer.h"

namespace neural_network {

NeuralNetworkTrainer::NeuralNetworkTrainer() = default;

Matrix NeuralNetworkTrainer::ForwardPropagate(
    const std::vector<float> &neuron_values) {

  Matrix temp;
  return temp;
}

Matrix
NeuralNetworkTrainer::BackPropagate(const std::vector<float> &neuron_values) {

  return Matrix{};
}

float NeuralNetworkTrainer::CalculatePointError(
    const std::vector<float> &expected_values,
    const std::vector<float> &actual_values) {

  return 0;
}

float NeuralNetworkTrainer::SigmoidActivator(float value) const {
  return 1 / (1 + std::exp(-value));
}

float NeuralNetworkTrainer::SigmoidActivatorDerivative(float value) const {
  return SigmoidActivator(value) * (1 - SigmoidActivator(value));
}

} // namespace neural_network