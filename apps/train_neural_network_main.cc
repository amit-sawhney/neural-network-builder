#include <iostream>

#include "core/model.h"

using neural_network::Matrix;
using neural_network::Model;

int main() {

  Model model({2, 1}, 0.01f);

  Matrix train_values{{0, 1}, {1, 0}};
  Matrix expected{{1}, {0}};

  model.Train(1000, train_values, expected);

  float ans = model.Predict({1, 0}).at(0);

  std::cout << ans << std::endl;

  return 0;
}