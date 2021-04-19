#pragma once

#include <vector>
namespace neural_network {

/**
 * Basic mathematical calculations associated with a Neural Network
 */
class Math {

public:
  /**
   * Calculates the value of the Sigmoid function at a cetain value
   *
   * @param value the value to input into the Sigmoid
   * @return the output of the Sigmoid function
   */
  static float CalculateSigmoid(float value);

  /**
   * Calculate the derivative of the sigmoid function
   *
   * @param value the value to calculate the derivative of
   * @param isSigmoidValue if the passed value is an output of the Sigmoid
   * @return the value of the derivative of Sigmoid at a certain value
   */
  static float CalculateSigmoidDerivative(float value,
                                          bool isSigmoidValue = false);

  /**
   * Determines the Mean Squared Error (MSE) between two sets of data
   *
   * @param expected_values the expected data values
   * @param actual_values the actual data values
   * @return the MSE
   */
  static float
  CalculateMeanSquaredError(const std::vector<float> &expected_values,
                            const std::vector<float> &actual_values);
};
} // namespace neural_network
