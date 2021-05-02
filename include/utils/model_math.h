#pragma once

#include <vector>

namespace neural_network {

/**
 * Calculates the value of the Sigmoid function at a certain value
 *
 * @param value the value to input into the Sigmoid
 * @return the output of the Sigmoid function
 */
float CalculateSigmoid(float value);

/**
 * Calculate the derivative of the sigmoid function
 *
 * @param value the value to calculate the derivative of
 * @param isSigmoidValue if the passed value is an output of the Sigmoid
 * @return the value of the derivative of Sigmoid at a certain value
 */
float CalculateSigmoidDerivative(float value, bool isSigmoidValue = false);

/**
 * Determines the Mean Squared Error (MSE) between two sets of data
 *
 * @param expected the expected data values
 * @param actual the actual data values
 * @return the MSE
 */
float CalculatePointError(float expected, float actual);

/**
 * Calculates the dot product between two vectors
 *
 * @param vector1 the first vector
 * @param vector2 the second vector
 * @return the sum of the products of each parallel element in the vectors
 */
float CalculateDotProduct(const std::vector<float> &vector1,
                          const std::vector<float> &vector2);

/**
 * Calculates the error associated with a layer
 *
 * @param actual_values the actual values that the network generated
 * @param expected_values the expected values that the network should have
 * generated
 * @return the output errors associated with the network layer
 */
std::vector<float>
CalculateErrorLayer(const std::vector<float> &actual_values,
                    const std::vector<float> &expected_values);
} // namespace neural_network
