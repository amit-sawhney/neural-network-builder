#include <catch2/catch.hpp>

#include "utils/model_math.h"

using neural_network::ModelMath;

const float kCalculationTolerance = 0.01f;

TEST_CASE("Sigmoid Calculations") {

  SECTION("Negative input") {
    float input = -1;

    float sigmoid = ModelMath::CalculateSigmoid(input);
    float expected_value = 0.2689f;

    REQUIRE(sigmoid == Approx(expected_value).epsilon(kCalculationTolerance));
  }

  SECTION("Zero input") {
    float input = 0;

    float sigmoid = ModelMath::CalculateSigmoid(input);
    float expected_value = 0.5f;

    REQUIRE(sigmoid == Approx(expected_value).epsilon(kCalculationTolerance));
  }

  SECTION("Positive input") {
    float input = 1;

    float sigmoid = ModelMath::CalculateSigmoid(input);
    float expected_value = 0.73106f;

    REQUIRE(sigmoid == Approx(expected_value).epsilon(kCalculationTolerance));
  }
}

TEST_CASE("Sigmoid Derivative Calculations") {

  SECTION("Negative input") {
    float input = -1;

    float sigmoid = ModelMath::CalculateSigmoidDerivative(input);
    float expected_value = 0.1967f;

    REQUIRE(sigmoid == Approx(expected_value).epsilon(kCalculationTolerance));
  }

  SECTION("Zero input") {
    float input = 0;

    float sigmoid = ModelMath::CalculateSigmoidDerivative(input);
    float expected_value = 0.25f;

    REQUIRE(sigmoid == Approx(expected_value).epsilon(kCalculationTolerance));
  }

  SECTION("Positive input") {
    float input = 1;

    float sigmoid = ModelMath::CalculateSigmoidDerivative(input);
    float expected_value = 0.196f;

    REQUIRE(sigmoid == Approx(expected_value).epsilon(kCalculationTolerance));
  }

  SECTION("Sigmoid input") {
    float input = ModelMath::CalculateSigmoid(1);
    bool isSigmoid = true;

    float sigmoid = ModelMath::CalculateSigmoidDerivative(input, isSigmoid);
    float expected_value = 0.1967f;

    REQUIRE(sigmoid == Approx(expected_value).epsilon(kCalculationTolerance));
  }
}
