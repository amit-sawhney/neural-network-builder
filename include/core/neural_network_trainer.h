#pragma once

#include <vector>
namespace neural_network {

typedef std::vector<std::vector<float>> Matrix;

class NeuralNetworkTrainer {

public:
  NeuralNetworkTrainer();

  Matrix ForwardPropagate(const std::vector<float> &neuron_values);

  Matrix BackPropagate(const std::vector<float> &neuron_values);

  float CalculatePointError(const std::vector<float> &expected_values,
                            const std::vector<float> &actual_values);

private:
  float SigmoidActivator(float value) const;

  float SigmoidActivatorDerivative(float value) const;
};
} // namespace neural_network