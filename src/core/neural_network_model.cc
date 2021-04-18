#include "core/neural_network_model.h"

namespace neural_network {

NeuralNetworkModel::NeuralNetworkModel() : learning_rate_(0), num_neurons_(0) {}

NeuralNetworkModel::NeuralNetworkModel(std::vector<size_t> neuron_layers,
                                       float learning_rate)
    : learning_rate_(learning_rate), neuron_layers_(std::move(neuron_layers)) {

  size_t neuron_count = 0;

  for (size_t neuron_layer : neuron_layers_) {
    neuron_count += neuron_layer;
  }

  num_neurons_ = neuron_count;

  InitializeModelWeights();
}

void NeuralNetworkModel::Clear() {}

void NeuralNetworkModel::Train(size_t epochs, const Matrix &input,
                               const Matrix &output) {}

std::ostream &operator<<(std::ostream &output,
                         const NeuralNetworkModel &model) {

  return output;
}

std::istream &operator>>(std::istream &input, NeuralNetworkModel &model) {

  return input;
}

void NeuralNetworkModel::InitializeModelWeights() {

  for (size_t layer = 0; layer < neuron_layers_.size() - 1; ++layer) {
    std::vector<float> weights;

    size_t weights_size = neuron_layers_[layer] * neuron_layers_[layer + 1];

    for (size_t weight = 0; weight < weights_size; ++weight) {
      weights.emplace_back(GenerateRandWeight());
    }

    model_weights_.emplace_back(weights);
  }
}

float NeuralNetworkModel::GenerateRandWeight() const {

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

Matrix NeuralNetworkModel::GetModelWeights() const { return model_weights_; }

} // namespace neural_network