#include "core/trainer.h"

namespace neural_network {

Trainer::Trainer() = default;

Trainer::Trainer(Matrix weights,
                                           std::vector<size_t> layer_sizes)
    : weights_(std::move(weights)), layer_sizes_(std::move(layer_sizes)) {}

Matrix Trainer::ForwardPropagate(const std::vector<float> &layer) {

  Matrix neurons{layer};

  for (size_t weight = 0; weight < weights_.size(); ++weight) {

    std::vector<float> layer_weights = weights_[weight];
    Matrix next_weights = CalculateNextLayerWeights(layer_weights, weight);

    std::vector<float> next_neurons =
        CalculateNextNeurons(neurons, next_weights, weight);

    neurons.emplace_back(next_neurons);
  }

  return neurons;
}

Matrix Trainer::BackPropagate(const std::vector<float> &expected_values,
                                    const Matrix &neuron_values) {




  return Matrix{};
}

Matrix Trainer::CalculateNextLayerWeights(
    const std::vector<float> &current_layer_weights,
    size_t current_weight_idx) const {

  Matrix next_layer_weights;

  for (size_t layer_idx = 0; layer_idx < layer_sizes_[current_weight_idx + 1];
       ++layer_idx) {

    std::vector<float> neuron_weights;

    for (size_t neuron = 0; neuron < layer_sizes_[current_weight_idx];
         ++neuron) {

      size_t next_neurons_size = layer_sizes_[current_weight_idx + 1];

      // Calculate the next neuron weight
      float next_weight =
          current_layer_weights[neuron * next_neurons_size + layer_idx];

      neuron_weights.emplace_back(next_weight);
    }

    next_layer_weights.emplace_back(neuron_weights);
  }

  return Matrix{};
}

std::vector<float>
Trainer::CalculateNextNeurons(const Matrix &neurons,
                                           const Matrix &weights,
                                           size_t current_weight_idx) const {
  std::vector<float> next_neurons;

  for (size_t layer = 0; layer < layer_sizes_[current_weight_idx + 1];
       ++layer) {

    float layer_dot_product = ModelMath::CalculateDotProduct(
        neurons.back(), weights[current_weight_idx]);

    // Apply activation function
    layer_dot_product = ModelMath::CalculateSigmoid(layer_dot_product);

    next_neurons.emplace_back(layer_dot_product);
  }

  return next_neurons;
}

} // namespace neural_network