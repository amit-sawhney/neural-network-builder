#include <catch2/catch.hpp>

#include "core/model.h"

using neural_network::Matrix;
using neural_network::Model;

TEST_CASE("Model default Constructor") {

  SECTION("Correct initialization") {
    Model model;

    REQUIRE(model.GetNumNeurons() == 0);
    REQUIRE(model.GetTrainer().GetWeights().empty());
  }
}

TEST_CASE("Regular constructor") {

  SECTION("Correct initialization") {

    Model model({2, 1}, 0.01f);

    REQUIRE(model.GetNumNeurons() == 3);
    REQUIRE(model.GetTrainer().GetLearningRate() == 0.01f);
    REQUIRE(model.GetTrainer().GetWeights().size() == 1);
  }
}

TEST_CASE("Predict") {

  SECTION("Correct values") {
    Model model({2, 1}, 0.01f);

    Matrix train_values{{0, 1}, {1, 0}};
    Matrix expected{{1}, {0}};

    model.Train(10, train_values, expected);

    float zero_prediction = model.Predict({1, 0}).at(0);

    REQUIRE(zero_prediction == Approx(0.40999f));
  }
}

TEST_CASE("Clear model") {

  SECTION("Model is cleared out") {

    Model model({2, 1}, 0.01f);

    model.Clear();

    REQUIRE(model.GetNumNeurons() == 0);
    REQUIRE(model.GetTrainer().GetLearningRate() == 0.0f);
    REQUIRE(model.GetTrainer().GetWeights().empty());
  }
}
