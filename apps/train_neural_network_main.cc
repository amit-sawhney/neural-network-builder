#include <iostream>
#include <numeric>

#include "core/model.h"

using neural_network::Matrix;
using neural_network::Model;

int main() {

  Model model({2, 1}, 0.01f);

  Matrix train_values{{0, 1}, {1, 0}};
  Matrix expected{{1}, {0}};

  model.Train(10, train_values, expected);

  return 0;
}