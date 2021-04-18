#include "core/neural_network_model.h"

namespace neural_network {

NeuralNetworkModel::NeuralNetworkModel() : learning_rate_(0){};

NeuralNetworkModel::NeuralNetworkModel(const std::vector<size_t> &neuron_layers,
                                       float learning_rate)
    : learning_rate_(learning_rate) {}

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

void NeuralNetworkModel::InitializeModelWeights() {}

} // namespace neural_network