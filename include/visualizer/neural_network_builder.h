#pragma once

#include <Windows.h>

#include "cinder/app/App.h"
#include "cinder/app/RendererGl.h"
#include "cinder/gl/gl.h"
#include "core/model.h"
#include "neuron.h"

namespace neural_network {

namespace visualizer {

typedef std::vector<Neuron> Layer;
typedef std::vector<Layer> Network;

/**
 * Cinder application that dynamically draws out a neural network architecture
 * with the ability to train and make predictions with the neural network
 */
class NeuralNetworkBuilderApp : public ci::app::App {

public:
  /**
   * Default Constructor to start the application
   */
  NeuralNetworkBuilderApp();

  /**
   * Draws the UI for the cinder application
   */
  void draw() override;

  /**
   * Listens for a file drop event and utilizes the passed data to train or
   * predict with the current neural network architecture
   *
   * @param event the file drop event
   */
  void fileDrop(ci::app::FileDropEvent event) override;

private:
  const std::vector<size_t> kLayerSizes{2, 1};

  /**
   * Builds out the UI for the neural network structure
   */
  void BuildNetworkStructure();

  /**
   * Determines the Neuron radius for the network layer
   *
   * @param x_pos the current x position of the neuron
   * @param height_interval the height interval between nodes in a layer
   * @return the neuron radius
   */
  float CalculateNeuronRadius(float x_pos, float height_interval) const;

  /**
   * Constructs a neuron structure based on where it is in the network and on
   * the screen
   *
   * @param current_layer the layer of the network the neuron is within
   * @param current_neuron the neuron in the specified layer
   * @param height_interval the height interval between the neurons
   * @param width_interval the width interval between the neurons
   * @return a dynamic Neuron
   */
  Neuron BuildDynamicNeuron(size_t current_layer, size_t current_neuron,
                            float height_interval, float width_interval) const;

  /**
   * Draws all of the lines between each of the nodes for the neural network
   */
  void DrawConnections() const;

  /**
   * Trains the model with a file input
   *
   * @param training_data the training data to train with
   */
  void TrainModel(std::ifstream *training_data);

  /**
   * Predicts the output for a set of data
   *
   * @param input_to_predict the input the neural network will use to predict
   */
  void Predict(std::ifstream *input_to_predict);

  /**
   * Updates the UI Neurons after training and predicting to show calculated
   * numbers
   *
   * @param output_values the output layer of the neural network
   */
  void
  UpdateVisualNeuralNetworkValues(const neural_network::Layer &output_values);

  float window_height_;
  float window_width_;
  Network network_;
  ci::Color neuron_color_;
  Model network_model_;
  float learning_rate_;
};
} // namespace visualizer

} // namespace neural_network
