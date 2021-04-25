#include "utils/model_math.h"

namespace neural_network {

float CalculateSigmoid(float value) {
  return 1 / (1 + std::exp(-value));
}

float CalculateSigmoidDerivative(float value, bool isSigmoidValue) {
  if (isSigmoidValue) {
    return value * (1 - value);
  }

  return CalculateSigmoid(value) * (1 - CalculateSigmoid(value));
}

float CalculatePointError(float expected, float actual) {

  float point_diff = expected - actual;

  return point_diff * CalculateSigmoidDerivative(actual);
}

float CalculateDotProduct(const std::vector<float> &vector1,
                                     const std::vector<float> &vector2) {

  if (vector1.size() != vector2.size()) {
    throw std::invalid_argument("Vector dimensions are not equal");
  }

  float total_sum = 0;

  for (size_t value_idx = 0; value_idx < vector1.size(); ++value_idx) {
    total_sum += (vector1[value_idx] * vector2[value_idx]);
  }

  return total_sum;
}

std::vector<float>
CalculateErrorLayer(const std::vector<float> &actual_values,
                               const std::vector<float> &expected_values) {

  if (actual_values.size() != expected_values.size()) {
    throw std::invalid_argument("Layer sizes are not equal");
  }

  std::vector<float> errors;
  for (size_t value_idx = 0; value_idx < expected_values.size(); ++value_idx) {
    float expected = expected_values[value_idx];
    float actual = actual_values[value_idx];

    float error = CalculatePointError(expected, actual);
    errors.emplace_back(error);
  }

  return errors;
}
} // namespace neural_network
