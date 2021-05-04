#pragma once

#include <vector>

#include "utils/model_math.h"

namespace neural_network {

typedef std::vector<float> Layer;
typedef std::vector<Layer> Matrix;

/**
 * Trains the Neural Network and handles the big 3 steps
 */
class Trainer {
public:
  /**
   * Default Constructor
   */
  Trainer();

  /**
   * Initializes the trainer with the necessary starting information
   *
   * @param weights a set of weights for each of the neurons in the network
   * @param layer_sizes the size of each layer in the neural network
   * @param learning_rate the learning rate (alpha) of the network
   */
  Trainer(Matrix weights, std::vector<size_t> layer_sizes, float learning_rate);

  /**
   * Executes the forward propagation on layer of the network
   *
   * @param layer the layer to forward propagate on
   * @return the values of all of the neurons after forward propagation
   */
  Matrix ForwardPropagate(const Layer &layer);

  /**
   * Executes the back propagation across a neural network
   *
   * @param output_errors the point error to calculate neuron value shifting
   * @param neuron_values the neuron values to backward propagate on
   */
  void BackPropagate(Matrix *output_errors, const Matrix &neuron_values);

  Matrix GetWeights() const;

  float GetLearningRate() const;

private:
  /**
   * Calculates the next set of layer weights on a forward propagation execution
   *
   * @param layer_weights the current layer weights
   * @param weight_idx the current neuron weight index that the forward
   * propagation execution is on
   * @return the new neuron values
   */
  Matrix CalculateNextForwardPropagationLayerWeights(const Layer &layer_weights,
                                                     size_t weight_idx) const;

  /**
   * Calculates the next set of layer weights on a backpropagation execution
   *
   * @param layer_weights the current layer weights the network is on
   * @param weight_idx the current neuron weight index that the forward
   * propagation execution is on
   * @return the new neuron values
   */
  Matrix CalculateNextBackPropagationLayerWeights(const Layer &layer_weights,
                                                  size_t weight_idx) const;

  /**
   * Calculates the next neuron values of the network
   *
   * @param neuron_values the current neurons values
   * @param weights the weights of the neurons
   * @param weight_idx the current neuron weight index
   * @return the new network layer of neurons
   */
  Layer CalculateNextNeurons(const Matrix &neuron_values, const Matrix &weights,
                             size_t weight_idx) const;

  /**
   * Calculates the new hidden layer_size neuron weights
   *
   * @param errors the errors associated with the layer_size
   * @param weights the weights of the neurons in the layer_size
   * @param neuron_values the values of the neurons in the layer_size
   * @param layer_size the size of the layer_size
   * @return the new network hidden layer of neurons
   */
  Layer CalculateHiddenLayerWeights(const Layer &errors, const Layer &weights,
                                    const Matrix &neuron_values,
                                    size_t layer_size) const;

  /**
   * Calculates a hidden layer's error values
   *
   * @param neuron_values the current neuron values
   * @param errors the errors associated with the neuron values
   * @param next_layer_weights the next layer's weights
   * @param layer_size the current layer size
   * @return the new network hidden layer errors
   */
  Layer CalculateHiddenLayerErrors(const Matrix &neuron_values,
                                   const Layer &errors,
                                   const Matrix &next_layer_weights,
                                   size_t layer_size) const;

  /**
   * Updates the trainer's weight matrix with a change of weights
   *
   * @param delta_weights the values to adjust the current weights by
   */
  void UpdateWeights(const Matrix &delta_weights);

  Matrix weights_;
  std::vector<size_t> layer_sizes_;
  float learning_rate_;
};
} // namespace neural_network