#pragma once

#include <core/trainer.h>
#include <vector>

namespace neural_network {

/**
 * Basic mathematical calculations associated with a Neural Network
 */
class ModelMath {

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
   * @param expected the expected data values
   * @param actual the actual data values
   * @return the MSE
   */
  static float CalculatePointError(float expected, float actual);

  /**
   * Calculates the dot product between two vectors
   *
   * @param vector1 the first vector
   * @param vector2 the second vector
   * @return the sum of the products of each parallel element in the vectors
   */
  static float CalculateDotProduct(const std::vector<float> &vector1,
                                   const std::vector<float> &vector2);
};
} // namespace neural_network
