#pragma once

#include "cinder/gl/gl.h"
#include <utility>

namespace neural_network {

namespace visualizer {

/**
 * Represents a UI neuron
 */
class Neuron {

public:
  /**
   * Default constructor for a Neuron
   */
  Neuron();

  /**
   * Initializes the values of the Neuron
   *
   * @param center_point the origin location of the neuron
   * @param radius the radius of the neuron
   * @param color the color of the neuron
   */
  Neuron(const glm::vec2 &center_point, float radius, const ci::Color &color);

  /**
   * Draws a neuron according to the instances specifications
   */
  void Draw() const;

  glm::vec2 GetInputConnectPoint() const;

  glm::vec2 GetOutputConnectPoint() const;

  void SetValue(float new_value);

private:
  glm::vec2 center_point_;
  glm::vec2 input_connect_point_;
  glm::vec2 output_connect_point_;
  float radius_;
  ci::Color color_;
  float value_;
};

} // namespace visualizer

} // namespace neural_network

