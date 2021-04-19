#pragma once

#include <vector>

#include "utils/model_math.h"

namespace neural_network {

typedef std::vector<std::vector<float>> Matrix;

class Trainer {

public:
  Trainer();

  Trainer(Matrix weights, std::vector<size_t> layer_sizes, float learning_rate);

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

  std::vector<float>
  CalculateErrorLayer(const std::vector<float> &actual_values,
                      const std::vector<float> &expected_values) const;

  std::vector<float>
  CalculateHiddenLayerWeights(const std::vector<float> &errors,
                              const std::vector<float> &current_weights,
                              const Matrix &current_neuron_values,
                              size_t current_layer) const;

  std::vector<float> CalculateHiddenLayerErrors(
      const Matrix &neuron_values, const std::vector<float> &errors,
      const Matrix &next_layer_weights, size_t current_layer) const;

  void UpdateWeights(const Matrix &delta_weights);

  Matrix weights_;
  std::vector<size_t> layer_sizes_;
  float learning_rate_;
};
} // namespace neural_network