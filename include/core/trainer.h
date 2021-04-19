#pragma once

#include <vector>

#include "utils/model_math.h"

namespace neural_network {

typedef std::vector<float> Layer;
typedef std::vector<Layer> Matrix;

class Trainer {

public:
  Trainer();

  Trainer(Matrix weights, std::vector<size_t> layer_sizes, float learning_rate);

  Matrix ForwardPropagate(const Layer &layer);

  Layer CalculateErrorLayer(const Layer &actual_values,
                            const Layer &expected_values) const;

  void BackPropagate(Matrix *output_errors, const Matrix &neuron_values);

private:
  Matrix CalculateNextLayerWeights(const Layer &current_layer_weights,
                                   size_t current_weight_idx) const;

  Layer CalculateNextNeurons(const Matrix &neurons, const Matrix &weights,
                             size_t current_weight_idx) const;

  Layer CalculateHiddenLayerWeights(const Layer &errors,
                                    const Layer &current_weights,
                                    const Matrix &current_neuron_values,
                                    size_t current_layer) const;

  Layer CalculateHiddenLayerErrors(const Matrix &neuron_values,
                                   const Layer &errors,
                                   const Matrix &next_layer_weights,
                                   size_t current_layer) const;

  void UpdateWeights(const Matrix &delta_weights);

  Matrix weights_;
  std::vector<size_t> layer_sizes_;
  float learning_rate_;
};
} // namespace neural_network