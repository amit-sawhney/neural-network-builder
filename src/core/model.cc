#include "core/model.h"

namespace neural_network {

Model::Model() : num_neurons_(0) {}

Model::Model(const std::vector<size_t> &neuron_layers, float learning_rate) {

  size_t neuron_count = 0;

  for (size_t neuron_layer : neuron_layers) {
    neuron_count += neuron_layer;
  }

  num_neurons_ = neuron_count;

  Matrix weights = InitializeModelWeights(neuron_layers);
  trainer_ = Trainer(weights, neuron_layers, learning_rate);
}

std::ostream &operator<<(std::ostream &output, const Model &model) {

  return output;
}

std::istream &operator>>(std::istream &input, Model &model) { return input; }

void Model::Clear() {
  num_neurons_ = 0;
  trainer_ = Trainer();
}

void Model::Train(size_t epochs, const Matrix &training_values,
                  const Matrix &expected_values) {

  // Determines how many times model will train on the data
  for (size_t epoch = 0; epoch < epochs; ++epoch) {

    for (size_t layer = 0; layer < training_values.size(); ++layer) {
      std::vector<float> layer_values = training_values[layer];

      Matrix neuron_values = trainer_.ForwardPropagate(layer_values);
      Matrix output_errors = {trainer_.CalculateErrorLayer(
          neuron_values.back(), expected_values[layer])};
      trainer_.BackPropagate(&output_errors, neuron_values);
    }
  }
}

Matrix Model::InitializeModelWeights(const std::vector<size_t> &layers) {

  Matrix weights;

  for (size_t layer = 0; layer < layers.size() - 1; ++layer) {
    std::vector<float> layer_weights;

    size_t weights_size = layers[layer] * layers[layer + 1];

    for (size_t weight = 0; weight < weights_size; ++weight) {
      layer_weights.emplace_back(GenerateRandomWeight());
    }

    weights.emplace_back(layer_weights);
  }

  return weights;
}

float Model::GenerateRandomWeight() const {

  // Choose weight value between 1 over the square root of the number of neurons
  // Note: Assumes the training dataset is standardized (X ~ N(0,1))
  float min = -1.0f / (float)std::sqrt(num_neurons_);
  float max = -min;

  // Code below derived from:
  // https://stackoverflow.com/questions/5289613/generate-random-float-between-two-floats/5289624
  float random_num = ((float)std::rand()) / (float)RAND_MAX;
  float range_diff = max - min;
  float random_inc = random_num * range_diff;

  return min + random_inc;
}

} // namespace neural_network