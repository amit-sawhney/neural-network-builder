#include "visualizer/neuron.h"

namespace neural_network {

namespace visualizer {

Neuron::Neuron() : radius_(0) {}

Neuron::Neuron(const glm::vec2 &center_point, float radius,
               const ci::Color &color)
    : center_point_(center_point), radius_(radius), color_(color) {

  glm::vec2 shift(radius, 0);
  input_connect_point_ = center_point_ - shift;
  output_connect_point_ = center_point_ + shift;
}

void Neuron::Draw() {
  ci::gl::color(color_);
  ci::gl::drawStrokedCircle(center_point_, radius_);
}

glm::vec2 Neuron::GetInputConnectPoint() const { return input_connect_point_; }

glm::vec2 Neuron::GetOutputConnectPoint() const {
  return output_connect_point_;
}
} // namespace visualizer

} // namespace neural_network
