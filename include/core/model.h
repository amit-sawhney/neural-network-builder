#pragma once

#include "trainer.h"
#include <vector>
namespace neural_network {

// Mathematical calculations and code were derived from below
// https://towardsdatascience.com/introduction-to-math-behind-neural-networks-e8b60dbbdeba
class NeuralNetworkModel {
public:
  /**
   * Default constructor
   */
  NeuralNetworkModel();

  /**
   * Builds a neural network with the number of layers and neurons in each layer
   * as specified
   *
   * @param neuron_layers the specified number of layers and neurons per layer
   */
  NeuralNetworkModel(std::vector<size_t> neuron_layers, float learning_rate);

  friend std::ostream &operator<<(std::ostream &output,
                                  const NeuralNetworkModel &model);

  friend std::istream &operator>>(std::istream &input,
                                  NeuralNetworkModel &model);

  void Train(size_t epochs, const Matrix &training_values, const Matrix &expected_values);

  /**
   * Clears the Neural Network of it's current values
   */
  void Clear();
  
  Matrix GetModelWeights() const;

private:
  void InitializeModelWeights();
  
  float GenerateRandWeight() const;

  Matrix model_weights_;
  std::vector<size_t> neuron_layers_;
  size_t num_neurons_;
  float learning_rate_;
  NeuralNetworkTrainer trainer_;
};

} // namespace neural_network