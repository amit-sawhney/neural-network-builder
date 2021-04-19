#include "utils/math.h"

namespace neural_network {

float Math::CalculateSigmoid(float value) { return 1 / (1 + std::exp(-value)); }

float Math::CalculateSigmoidDerivative(float value, bool isSigmoidValue) {
  if (isSigmoidValue) {
    return value * (1 - value);
  }

  return CalculateSigmoid(value) * (1 - CalculateSigmoid(value));
}

float Math::CalculateMeanSquaredError(const std::vector<float> &expected_values,
                                      const std::vector<float> &actual_values) {

  if (expected_values.size() != actual_values.size()) {
    throw std::invalid_argument("Values dimensions are not equal");
  }

  float total_error = 0;
  size_t num_values = expected_values.size();

  for (size_t value = 0; value < num_values; ++value) {

    float value_diff = actual_values[value] - expected_values[value];
    total_error += std::pow(value_diff, 2);
  }

  return total_error / num_values;
}
} // namespace neural_network