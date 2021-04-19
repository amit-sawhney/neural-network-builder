#pragma once

#include "trainer.h"
#include <vector>
namespace neural_network {

// Mathematical calculations and code were derived from below
// https://towardsdatascience.com/introduction-to-math-behind-neural-networks-e8b60dbbdeba
class Model {
public:
  /**
   * Default constructor
   */
  Model();

  /**
   * Builds a neural network with the number of layers and neurons in each layer
   * as specified
   *
   * @param neuron_layers the specified number of layers and neurons per layer
   */
  Model(const std::vector<size_t> &neuron_layers, float learning_rate);

  friend std::ostream &operator<<(std::ostream &output, const Model &model);

  friend std::istream &operator>>(std::istream &input, Model &model);

  void Train(size_t epochs, const Matrix &training_values,
             const Matrix &expected_values);

  Layer Predict(const Layer &input_layer);

  /**
   * Clears the Neural Network of it's current values
   */
  void Clear();

private:
  Matrix InitializeModelWeights(const std::vector<size_t> &layers);

  float GenerateRandomWeight() const;

  size_t num_neurons_;
  Trainer trainer_;
};

} // namespace neural_network