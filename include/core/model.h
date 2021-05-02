#pragma once

#include <vector>

#include "trainer.h"

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

  /**
   * Trains the neural network on a certain set of data
   *
   * @param epochs the number of times the neural network will see the data
   * @param training_values the set training data to train on
   * @param expected_values the expected output of the training values
   */
  void Train(size_t epochs, const Matrix &training_values,
             const Matrix &expected_values);

  Layer Predict(const Layer &input_layer);

  /**
   * Clears the Neural Network of it's current values
   */
  void Clear();

  Trainer GetTrainer() const;

  size_t GetNumNeurons() const;

private:
  /**
   * Initializes the neuron weights of the neural network
   *
   * @param layers the size of each layer
   * @return the randomly generated weights
   */
  Matrix InitializeModelWeights(const std::vector<size_t> &layers);

  /**
   * Generates a random weight for the a single neuron of the network
   *
   * @return the random weight
   */
  float GenerateRandomWeight() const;

  size_t num_neurons_;
  Trainer trainer_;
};

} // namespace neural_network