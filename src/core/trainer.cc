#include "core/trainer.h"

namespace neural_network {

Trainer::Trainer() : learning_rate_(0) {}

Trainer::Trainer(Matrix weights, std::vector<size_t> layer_sizes,
                 float learning_rate)
    : weights_(std::move(weights)), layer_sizes_(std::move(layer_sizes)),
      learning_rate_(learning_rate) {}

Matrix Trainer::ForwardPropagate(const Layer &layer) {

  if (layer.size() != layer_sizes_.at(0)) {
    throw std::invalid_argument("Cannot propagate layer. Invalid size");
  }

  Matrix neurons{layer};

  for (size_t weight = 0; weight < weights_.size(); ++weight) {

    Layer layer_weights = weights_[weight];
    Matrix next_weights =
        CalculateNextForwardPropagationLayerWeights(layer_weights, weight);

    Layer next_neurons = CalculateNextNeurons(neurons, next_weights, weight);

    neurons.emplace_back(next_neurons);
  }

  return neurons;
}

void Trainer::BackPropagate(Matrix *output_errors,
                            const Matrix &neuron_values) {

  size_t penultimate_layer = neuron_values.size() - 2;
  Matrix weight_changes;

  // Go across matrix backwards
  for (int layer = penultimate_layer; layer >= 0; --layer) {
    Layer layer_weights = weights_[layer];

    Matrix next_layer_weights =
        CalculateNextBackPropagationLayerWeights(layer_weights, layer);

    Layer errors = output_errors->back();
    Layer hidden_layer_delta_weights = CalculateHiddenLayerWeights(
        errors, layer_weights, neuron_values, layer);

    Layer hidden_layer_errors = CalculateHiddenLayerErrors(
        neuron_values, errors, next_layer_weights, layer);

    output_errors->emplace_back(hidden_layer_errors);
    weight_changes.emplace_back(hidden_layer_delta_weights);
  }

  UpdateWeights(weight_changes);
}

Layer Trainer::CalculateHiddenLayerErrors(const Matrix &neuron_values,
                                          const Layer &errors,
                                          const Matrix &next_layer_weights,
                                          size_t layer_size) const {
  Layer layer_errors;

  for (size_t layer_idx = 0; layer_idx < layer_sizes_[layer_size];
       ++layer_idx) {

    // Multiply errors and layer
    Layer next_layer = next_layer_weights[layer_idx];
    float product = CalculateDotProduct(errors, next_layer);

    float neuron_value = neuron_values[layer_size][layer_idx];
    product *= CalculateSigmoidDerivative(neuron_value);

    layer_errors.emplace_back(product);
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
                                           const Layer &weights,
                                           const Matrix &neuron_values,
                                           size_t layer_size) const {

  Layer delta_weights;

  for (size_t weight_idx = 0; weight_idx < weights.size(); ++weight_idx) {
    size_t prev_neuron = weight_idx / layer_sizes_[layer_size + 1];
    size_t new_neuron_idx = weight_idx % layer_sizes_[layer_size + 1];

    float new_neuron_error = errors[new_neuron_idx];
    float new_neuron_weight =
        new_neuron_error * neuron_values[layer_size][prev_neuron];

    // Adjust step size but learning rate
    new_neuron_weight *= learning_rate_;

    delta_weights.emplace_back(new_neuron_weight);
  }

  return delta_weights;
}

Matrix
Trainer::CalculateNextBackPropagationLayerWeights(const Layer &layer_weights,
                                                  size_t weight_idx) const {
  Matrix next_layer_weights;

  for (size_t neuron = 0; neuron < layer_sizes_[weight_idx]; ++neuron) {

    Layer neuron_weights;

    for (size_t layer_idx = 0; layer_idx < layer_sizes_[weight_idx + 1];
         ++layer_idx) {

      size_t next_neurons_size = layer_sizes_[weight_idx + 1];

      // Calculate the next neuron weight
      float next_weight = layer_weights[neuron * next_neurons_size + layer_idx];

      neuron_weights.emplace_back(next_weight);
    }

    next_layer_weights.emplace_back(neuron_weights);
  }

  return next_layer_weights;
}

Matrix
Trainer::CalculateNextForwardPropagationLayerWeights(const Layer &layer_weights,
                                                     size_t weight_idx) const {

  Matrix next_layer_weights;

  for (size_t layer_idx = 0; layer_idx < layer_sizes_[weight_idx + 1];
       ++layer_idx) {

    Layer neuron_weights;

    for (size_t neuron = 0; neuron < layer_sizes_[weight_idx]; ++neuron) {
      size_t next_neurons_size = layer_sizes_[weight_idx + 1];

      // Calculate the next neuron weight
      float next_weight = layer_weights[neuron * next_neurons_size + layer_idx];
      neuron_weights.emplace_back(next_weight);
    }

    next_layer_weights.emplace_back(neuron_weights);
  }

  return next_layer_weights;
}

Layer Trainer::CalculateNextNeurons(const Matrix &neuron_values,
                                    const Matrix &weights,
                                    size_t weight_idx) const {
  Layer next_neurons;

  for (size_t layer = 0; layer < layer_sizes_[weight_idx + 1]; ++layer) {

    float layer_dot_product = CalculateDotProduct(
        neuron_values.back(), weights[weight_idx]);

    // Apply activation function
    layer_dot_product = CalculateSigmoid(layer_dot_product);

    next_neurons.emplace_back(layer_dot_product);
  }

  return next_neurons;
}

float Trainer::GetLearningRate() const { return learning_rate_; }

Matrix Trainer::GetWeights() const { return weights_; }

} // namespace neural_network