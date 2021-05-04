#include "visualizer/neural_network_builder.h"

namespace neural_network {
namespace visualizer {

NeuralNetworkBuilderApp::NeuralNetworkBuilderApp()
    : neuron_color_(ci::Color("white")), learning_rate_(0.01f),
      layer_sizes({1}) {

  window_width_ = float(GetSystemMetrics(SM_CXFULLSCREEN));
  window_height_ = float(GetSystemMetrics(SM_CYFULLSCREEN));

  ci::app::setWindowSize(int(window_width_), int(window_height_));

  BuildNetworkStructure();
  network_model_ = Model(layer_sizes, learning_rate_);
}

void NeuralNetworkBuilderApp::fileDrop(ci::app::FileDropEvent event) {

  std::ifstream input_file;
  input_file.open(event.getFile(0));

  std::string current_line;
  std::getline(input_file, current_line);

  std::string id = current_line;

  if (id == "TRAIN") {
    TrainModel(&input_file);
  } else if (id == "PREDICT") {
    Predict(&input_file);
  } else {
    throw std::invalid_argument("Invalid file type");
  }
}

void NeuralNetworkBuilderApp::keyDown(ci::app::KeyEvent event) {
  switch (event.getCode()) {
  case ci::app::KeyEvent::KEY_DOWN:
    if (layer_sizes.back() > 1) {
      network_.clear();
      --layer_sizes.back();
      BuildNetworkStructure();
    }
    break;
  case ci::app::KeyEvent::KEY_UP:
    network_.clear();
    ++layer_sizes.back();
    BuildNetworkStructure();
    break;
  case ci::app::KeyEvent::KEY_RIGHT:
    network_.clear();
    layer_sizes.push_back(1);
    network_model_ = Model(layer_sizes, learning_rate_);
    BuildNetworkStructure();
    break;
  case ci::app::KeyEvent::KEY_LEFT:
    if (layer_sizes.size() > 1) {
      network_.clear();
      layer_sizes.pop_back();
      network_model_ = Model(layer_sizes, learning_rate_);
      BuildNetworkStructure();
    }
    break;
  }
}

void NeuralNetworkBuilderApp::TrainModel(std::ifstream *training_data) {

  std::string current_line;
  Matrix expected_values;
  Matrix training_values;

  while (std::getline(*training_data, current_line)) {
    std::stringstream line_stream(current_line);
    float value;

    line_stream >> value;
    expected_values.push_back({value});

    std::vector<float> training_set;

    while (line_stream >> value) {
      training_set.push_back(value);
    }

    training_values.push_back(training_set);
  }

  network_model_.Train(10, training_values, expected_values);
}

void NeuralNetworkBuilderApp::Predict(std::ifstream *input_to_predict) {

  std::string current_line;
  std::getline(*input_to_predict, current_line);

  std::stringstream line_stream(current_line);
  float value;

  std::vector<float> predict_data;

  while (line_stream >> value) {
    predict_data.push_back(value);
  }

  neural_network::Layer output = network_model_.Predict(predict_data);

  UpdateVisualNeuralNetworkValues(output);
}

void NeuralNetworkBuilderApp::UpdateVisualNeuralNetworkValues(
    const neural_network::Layer &output_values) {

  for (size_t neuron = 0; neuron < output_values.size(); ++neuron) {
    network_[network_.size() - 1][neuron].SetValue(output_values.at(neuron));
  }
}

void NeuralNetworkBuilderApp::BuildNetworkStructure() {

  if (layer_sizes.empty()) {
    return;
  }

  network_.clear();

  float x_interval = window_width_ / (float(layer_sizes.size()) + 1);

  for (size_t layer = 0; layer < layer_sizes.size(); ++layer) {

    Layer network_layer;
    size_t layer_size = layer_sizes[layer];

    if (layer_size == 0) {
      throw std::invalid_argument("Invalid layer size");
    }

    float y_interval = window_height_ / (float(layer_size) + 1);

    for (size_t neuron = 0; neuron < layer_size; ++neuron) {
      Neuron new_neuron =
          BuildDynamicNeuron(layer, neuron, y_interval, x_interval);

      network_layer.emplace_back(new_neuron);
    }

    network_.emplace_back(network_layer);
  }
}

void NeuralNetworkBuilderApp::draw() {
  ci::gl::clear();

  for (const Layer &layer : network_) {
    for (const Neuron &neuron : layer) {
      neuron.Draw();
    }
  }

  DrawConnections();
}

void NeuralNetworkBuilderApp::DrawConnections() const {

  for (int layer = 0; layer < int(network_.size()) - 1; ++layer) {

    Layer current_layer = network_[layer];
    Layer next_layer = network_[layer + 1];

    for (const Neuron &neuron : current_layer) {
      for (const Neuron &next_neuron : next_layer) {
        ci::gl::drawLine(neuron.GetOutputConnectPoint(),
                         next_neuron.GetInputConnectPoint());
      }
    }
  }
}

float NeuralNetworkBuilderApp::CalculateNeuronRadius(
    float x_pos, float height_interval) const {

  float neuron_margin = 10.0f;
  float remaining_width = window_width_ - x_pos;

  // Determine whether there is less height or width available for the neuron
  float diameter = std::min(height_interval / 2, remaining_width);

  return diameter / 2 - neuron_margin;
}

Neuron NeuralNetworkBuilderApp::BuildDynamicNeuron(size_t current_layer,
                                                   size_t current_neuron,
                                                   float height_interval,
                                                   float width_interval) const {

  glm::vec2 center(width_interval * (float(current_layer + 1)),
                   height_interval * (float(current_neuron + 1)));

  float neuron_radius = CalculateNeuronRadius(center.x, height_interval);
  Neuron new_neuron(center, neuron_radius, neuron_color_);

  return new_neuron;
}

} // namespace visualizer

} // namespace neural_network
