#pragma once

#include "cinder/gl/gl.h"
#include <utility>

namespace neural_network {

namespace visualizer {

class Neuron {

public:
  Neuron();

  Neuron(const glm::vec2& center_point, float radius, const ci::Color& color);

  void Draw();

  glm::vec2 GetInputConnectPoint() const;

  glm::vec2 GetOutputConnectPoint() const;

private:
  glm::vec2 center_point_;
  glm::vec2 input_connect_point_;
  glm::vec2 output_connect_point_;
  float radius_;
  ci::Color color_;
};
} // namespace visualizer
} // namespace neural_network
