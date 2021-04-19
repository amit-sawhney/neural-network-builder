#include <iostream>

#include "core/model.h"

using neural_network::Matrix;
using neural_network::Model;

int main() {

  Model model({1, 2, 1}, 0.01f);

  return 0;
}