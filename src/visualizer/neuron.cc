#include "visualizer/neuron.h"

namespace neural_network {

namespace visualizer {

Neuron::Neuron() : radius_(0), value_(0) {}

Neuron::Neuron(const glm::vec2 &center_point, float radius,
               const ci::Color &color)
    : center_point_(center_point), radius_(radius), color_(color), value_(0) {

  glm::vec2 shift(radius, 0);
  input_connect_point_ = center_point_ - shift;
  output_connect_point_ = center_point_ + shift;
}

void Neuron::Draw() const {
  ci::gl::color(color_);
  ci::gl::drawStrokedCircle(center_point_, radius_);

  if (value_ > 0) {
    // const addresses memory leak in font
    const ci::Font text_size("Text Size", 30);
    ci::Color text_color("white");
    std::string rounded_text = std::to_string(value_).substr(0, 4);

    ci::gl::drawStringCentered(rounded_text, center_point_, text_color,
                               text_size);
  }
}

glm::vec2 Neuron::GetInputConnectPoint() const { return input_connect_point_; }

glm::vec2 Neuron::GetOutputConnectPoint() const {
  return output_connect_point_;
}

void Neuron::SetValue(float new_value) { value_ = new_value; }

} // namespace visualizer

} // namespace neural_network