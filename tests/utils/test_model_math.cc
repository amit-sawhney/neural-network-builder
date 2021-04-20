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

TEST_CASE("Point Error Calculations") {

  SECTION("Negative values") {
    float expected_input = -1;
    float actual_input = -2;

    float error = ModelMath::CalculatePointError(expected_input, actual_input);
    float expected_error = 0.105f;

    REQUIRE(error == Approx(expected_error).epsilon(kCalculationTolerance));
  }

  SECTION("Negative and positive value") {
    float expected_input = -1;
    float actual_input = 1;

    float error = ModelMath::CalculatePointError(expected_input, actual_input);
    float expected_error = -0.3932f;

    REQUIRE(error == Approx(expected_error).epsilon(kCalculationTolerance));
  }

  SECTION("Positive and negative value") {
    float expected_input = 1;
    float actual_input = -1;

    float error = ModelMath::CalculatePointError(expected_input, actual_input);
    float expected_error = 0.3932f;

    REQUIRE(error == Approx(expected_error).epsilon(kCalculationTolerance));
  }

  SECTION("Positive and positive value") {
    float expected_input = 1;
    float actual_input = 2;

    float error = ModelMath::CalculatePointError(expected_input, actual_input);
    float expected_error = -0.105f;

    REQUIRE(error == Approx(expected_error).epsilon(kCalculationTolerance));
  }

  SECTION("No error input") {
    float expected_input = 1;
    float actual_input = 1;

    float error = ModelMath::CalculatePointError(expected_input, actual_input);
    float expected_error = 0;

    REQUIRE(error == Approx(expected_error).epsilon(kCalculationTolerance));
  }
}
