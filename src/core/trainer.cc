#include "core/trainer.h"

namespace neural_network {

Trainer::Trainer() : learning_rate_(0) {}

Trainer::Trainer(Matrix weights, std::vector<size_t> layer_sizes,
                 float learning_rate)
    : weights_(std::move(weights)), layer_sizes_(std::move(layer_sizes)),
      learning_rate_(learning_rate) {}

Matrix Trainer::ForwardPropagate(const Layer &layer) {

  Matrix neurons{layer};

  for (size_t weight = 0; weight < weights_.size(); ++weight) {

    Layer layer_weights = weights_[weight];
    Matrix next_weights = CalculateNextLayerWeights(layer_weights, weight);

    Layer next_neurons = CalculateNextNeurons(neurons, next_weights, weight);

    neurons.emplace_back(next_neurons);
  }

  return neurons;
}

void Trainer::BackPropagate(Matrix *output_errors,
                            const Matrix &neuron_values) {

  size_t penultimate_layer = neuron_values.size() - 2;

  Matrix total_weight_changes;

  for (size_t layer = penultimate_layer; layer >= 0; --layer) {
    Layer layer_weights = weights_[layer];

    Matrix next_layer_weights = CalculateNextLayerWeights(layer_weights, layer);

    Layer errors = output_errors->back();
    Layer hidden_layer_delta_weights = CalculateHiddenLayerWeights(
        errors, layer_weights, neuron_values, layer);

    Layer hidden_layer_errors = CalculateHiddenLayerErrors(
        neuron_values, errors, next_layer_weights, layer);

    output_errors->push_back(hidden_layer_errors);
    total_weight_changes.push_back(hidden_layer_delta_weights);
  }

  UpdateWeights(total_weight_changes);
}

Layer Trainer::CalculateHiddenLayerErrors(const Matrix &neuron_values,
                                          const Layer &errors,
                                          const Matrix &next_layer_weights,
                                          size_t current_layer) const {
  Layer layer_errors;

  for (size_t layer_idx = 0; layer_idx < layer_sizes_[current_layer];
       ++layer_idx) {

    // Multiply the corresponding row and column
    Layer next_layer = next_layer_weights[layer_idx];
    float product = ModelMath::CalculateDotProduct(errors, next_layer);

    float neuron_value = neuron_values[current_layer][layer_idx];
    product *= ModelMath::CalculateSigmoidDerivative(neuron_value);

    layer_errors.push_back(product);
  }

  return layer_errors;
}

void Trainer::UpdateWeights(const Matrix &delta_weights) {

  for (size_t row = 0; row < weights_.size(); ++row) {
    for (size_t col = 0; col < weights_.size(); ++col) {

      // Update the values of the weight matrix with the correct deltas
      weights_[row][col] += delta_weights[delta_weights.size() - 1 - row][col];
    }
  }
}

Layer Trainer::CalculateHiddenLayerWeights(const Layer &errors,
                                           const Layer &current_weights,
                                           const Matrix &current_neuron_values,
                                           size_t current_layer) const {

  Layer delta_weights;

  for (size_t weight_idx = 0; weight_idx < current_weights.size();
       ++weight_idx) {
    size_t prev_neuron = weight_idx / layer_sizes_[current_layer + 1];
    size_t new_neuron_idx = weight_idx % layer_sizes_[current_layer + 1];

    float new_neuron_error = errors[new_neuron_idx];
    float new_neuron_weight =
        new_neuron_error * current_neuron_values[current_layer][prev_neuron];

    // Adjust step size but learning rate
    new_neuron_weight *= learning_rate_;

    delta_weights.emplace_back(new_neuron_weight);
  }

  return delta_weights;
}

Layer Trainer::CalculateErrorLayer(const Layer &actual_values,
                                   const Layer &expected_values) const {

  Layer errors;
  for (size_t value_idx = 0; value_idx < expected_values.size(); ++value_idx) {
    float expected = expected_values[value_idx];
    float actual = actual_values[value_idx];

    float error = ModelMath::CalculatePointError(expected, actual);
    errors.push_back(error);
  }

  return errors;
}

Matrix Trainer::CalculateNextLayerWeights(const Layer &current_layer_weights,
                                          size_t current_weight_idx) const {

  Matrix next_layer_weights;

  for (size_t layer_idx = 0; layer_idx < layer_sizes_[current_weight_idx + 1];
       ++layer_idx) {

    Layer neuron_weights;

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

Layer Trainer::CalculateNextNeurons(const Matrix &neurons,
                                    const Matrix &weights,
                                    size_t current_weight_idx) const {
  Layer next_neurons;

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