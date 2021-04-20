#include <catch2/catch.hpp>

#include "utils/model_math.h"

using neural_network::ModelMath;

const float kCalculationTolerance = 0.1f;

TEST_CASE("Sigmoid Calculations") {

  SECTION("Negative value") {
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

  SECTION("Positive value") {
    float input = 1;

    float sigmoid = ModelMath::CalculateSigmoid(input);
    float expected_value = 0.73106f;

    REQUIRE(sigmoid == Approx(expected_value).epsilon(kCalculationTolerance));
  }
}
