#include "visualizer/neuron.h"
#include <catch2/catch.hpp>

using neural_network::visualizer::Neuron;

TEST_CASE("Neuron initialization") {

  SECTION("Input connection") {
    glm::vec2 center(10, 10);
    float radius = 5;
    ci::Color color("white");

    Neuron neuron(center, radius, color);

    REQUIRE(neuron.GetInputConnectPoint() == glm::vec2(5, 10));
  }

  SECTION("Output Connection") {
    glm::vec2 center(10, 10);
    float radius = 5;
    ci::Color color("white");

    Neuron neuron(center, radius, color);

    REQUIRE(neuron.GetOutputConnectPoint() == glm::vec2(15, 10));
  }
}
