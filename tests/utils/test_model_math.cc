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

  SECTION("Zero error input") {
    float expected_input = 1;
    float actual_input = 1;

    float error = ModelMath::CalculatePointError(expected_input, actual_input);
    float expected_error = 0;

    REQUIRE(error == Approx(expected_error).epsilon(kCalculationTolerance));
  }
}

TEST_CASE("Dot Product Calculations") {

  SECTION("Invalid dimensions") {
    std::vector<float> vector1;
    std::vector<float> vector2{1.0f};

    REQUIRE_THROWS(ModelMath::CalculateDotProduct(vector1, vector2));
  }

  SECTION("Positive values in both vectors") {
    std::vector<float> vector1{1, 2};
    std::vector<float> vector2{1, 2};

    float product = ModelMath::CalculateDotProduct(vector1, vector2);
    float expected_product = 5.0f;

    REQUIRE(product == Approx(expected_product).epsilon(kCalculationTolerance));
  }

  SECTION("Negative values in both vectors") {
    std::vector<float> vector1{-1, -2};
    std::vector<float> vector2{-1, -2};

    float product = ModelMath::CalculateDotProduct(vector1, vector2);
    float expected_product = 5.0f;

    REQUIRE(product == Approx(expected_product).epsilon(kCalculationTolerance));
  }

  SECTION("Mixed values in both vectors") {
    std::vector<float> vector1{-1, -2};
    std::vector<float> vector2{1, 2};

    float product = ModelMath::CalculateDotProduct(vector1, vector2);
    float expected_product = -5.0f;

    REQUIRE(product == Approx(expected_product).epsilon(kCalculationTolerance));
  }

  SECTION("Orthogonal vectors") {
    std::vector<float> vector1{1, 1};
    std::vector<float> vector2{-1, 1};

    float product = ModelMath::CalculateDotProduct(vector1, vector2);
    float expected_product = 0.0f;

    REQUIRE(product == Approx(expected_product).epsilon(kCalculationTolerance));
  }

  SECTION("Order insensitive") {
    std::vector<float> vector1{-1, -2};
    std::vector<float> vector2{1, 2};

    float product1 = ModelMath::CalculateDotProduct(vector1, vector2);
    float product2 = ModelMath::CalculateDotProduct(vector2, vector1);

    REQUIRE(product1 == product2);
  }
}

TEST_CASE("Error Layer Calculations") {

  SECTION("Invalid layer sizes") {
    std::vector<float> empty_layer{};
    std::vector<float> filled_layer{0};

    REQUIRE_THROWS(ModelMath::CalculateErrorLayer(empty_layer, filled_layer));
  }

  SECTION("Correct error values") {
    std::vector<float> empty_layer{1, 0, -1};
    std::vector<float> filled_layer{-2, 1, 0};

    std::vector<float> error_layer =
        ModelMath::CalculateErrorLayer(empty_layer, filled_layer);
    std::vector<float> expected_error_layer{-0.58984f, 0.25f, 0.19661f};

    REQUIRE(expected_error_layer.at(0) ==
            Approx(error_layer.at(0)).epsilon(kCalculationTolerance));
    REQUIRE(expected_error_layer.at(1) ==
            Approx(error_layer.at(1)).epsilon(kCalculationTolerance));
    REQUIRE(expected_error_layer.at(2) ==
            Approx(error_layer.at(2)).epsilon(kCalculationTolerance));
  }
}
