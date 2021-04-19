#pragma once

#include <vector>

#include "utils/model_math.h"

namespace neural_network {

typedef std::vector<std::vector<float>> Matrix;

class NeuralNetworkTrainer {

public:
  NeuralNetworkTrainer();

  NeuralNetworkTrainer(Matrix weights, std::vector<size_t> layer_sizes);

  Matrix ForwardPropagate(const std::vector<float> &layer);

  Matrix BackPropagate(const std::vector<float> &expected_values,
                       const Matrix &neuron_values);

private:
  Matrix
  CalculateNextLayerWeights(const std::vector<float> &current_layer_weights,
                            size_t current_weight_idx) const;

  std::vector<float> CalculateNextNeurons(const Matrix &neurons,
                                          const Matrix &weights,
                                          size_t current_weight_idx) const;

  Matrix weights_;
  std::vector<size_t> layer_sizes_;
};
} // namespace neural_network