#include <catch2/catch.hpp>
#include <iostream>

#include "core/trainer.h"

using neural_network::Layer;
using neural_network::Matrix;
using neural_network::Trainer;

TEST_CASE("Default Constructor") {
  Trainer trainer;

  SECTION("Learning rate is 0") { REQUIRE(trainer.GetLearningRate() == 0); }

  SECTION("Weights are empty") { REQUIRE(trainer.GetWeights().empty()); }
}

TEST_CASE("Forward Propagation calculations") {

  float learning_rate = 0.01f;

  SECTION("Invalid layer and weight sizes") {
    Matrix test_weights{{1, 1}};
    std::vector<size_t> test_layers{2, 1};

    Trainer trainer(test_weights, test_layers, learning_rate);

    REQUIRE_THROWS(trainer.ForwardPropagate({1, 1, 1}));
  }

  SECTION("Correct forward propagation calculation") {
    Matrix test_weights{{1.0f, 1.0f}};
    std::vector<size_t> test_layers{2, 1};

    Trainer trainer(test_weights, test_layers, learning_rate);

    Layer layer_to_propagate{1.0f, 2.0f};
    Matrix weights = trainer.ForwardPropagate(layer_to_propagate);

    std::vector<float> weights1{1, 2};
    std::vector<float> weights2{0.95257f};

    REQUIRE(weights.size() == 2);
    REQUIRE(weights.at(0) == weights1);
    REQUIRE(weights.at(1).at(0) == Approx(weights2.at(0)));
  }
}

TEST_CASE("Backpropagation calculations") {

  float learning_rate = 0.01f;

  SECTION("Correct backpropagation values") {
    Matrix test_weights{{1, 1}};
    std::vector<size_t> test_layers{2, 1};

    Trainer trainer(test_weights, test_layers, learning_rate);

    Matrix neuron_values = trainer.ForwardPropagate({1, 0});

    Layer output_layer = neuron_values.back();
    Matrix output_errors{
        neural_network::CalculateErrorLayer(output_layer, {1.0f})};

    trainer.BackPropagate(&output_errors, neuron_values);

    Matrix weights = trainer.GetWeights();

    REQUIRE(weights.size() == 1);
    REQUIRE(weights.at(0).at(0) == Approx(1.00059f));
    REQUIRE(weights.at(0).at(1) == Approx(1.0f));
  }
}