#include "core/neural_network_trainer.h"

namespace neural_network {

NeuralNetworkTrainer::NeuralNetworkTrainer() = default;

NeuralNetworkTrainer::NeuralNetworkTrainer(Matrix weights,
                                           std::vector<size_t> layer_sizes)
    : weights_(std::move(weights)), layer_sizes_(std::move(layer_sizes)) {}

Matrix NeuralNetworkTrainer::ForwardPropagate(
    const std::vector<float> &neuron_values) {

  return Matrix{};
}

Matrix
NeuralNetworkTrainer::BackPropagate(const std::vector<float> &expected_values,
                                    const Matrix &neuron_values) {

  return Matrix{};
}

} // namespace neural_network