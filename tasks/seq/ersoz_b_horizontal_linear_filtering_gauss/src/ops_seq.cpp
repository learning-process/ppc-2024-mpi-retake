#include "seq/ersoz_b_horizontal_linear_filtering_gauss/include/ops_seq.hpp"

#include <cmath>
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif
#include <cstdint>
#include <utility>
#include <vector>

namespace {

inline double GaussianFunction(int i, int j, double sigma) {
  return 1.0 / (2 * M_PI * sigma * sigma) * exp(-(((i * i)) + ((j * j))) / (2 * sigma * sigma));
}

std::vector<std::vector<char>> SequentialGaussianFilter(const std::vector<std::vector<char>>& image, double sigma) {
  int y_dim = static_cast<int>(image.size());
  int x_dim = static_cast<int>(image[0].size());
  std::vector<std::vector<char>> res;
  for (int y = 1; y < y_dim - 1; y++) {
    std::vector<char> line;
    for (int x = 1; x < x_dim - 1; x++) {
      double brightness = 0;
      for (int i = -1; i <= 1; i++) {
        for (int j = -1; j <= 1; j++) {
          brightness += GaussianFunction(i, j, sigma) * static_cast<int>(image[y + i][x + j]);
        }
      }
      line.emplace_back(static_cast<char>(brightness));
    }
    res.emplace_back(std::move(line));
  }
  return res;
}

}  // namespace

bool ersoz_b_test_task_seq::TestTaskSequential::PreProcessingImpl() {
  unsigned int input_size = task_data->inputs_count[0];
  img_size_ = static_cast<int>(std::sqrt(input_size));
  uint8_t* in_ptr = task_data->inputs[0];
  std::vector<char> flat(in_ptr, in_ptr + input_size);
  input_image_.resize(img_size_);
  for (int i = 0; i < img_size_; i++) {
    input_image_[i] = std::vector<char>(flat.begin() + i * img_size_, flat.begin() + ((i + 1) * img_size_));
  }
  output_image_.resize(img_size_ - 2);
  for (int i = 0; i < img_size_ - 2; i++) {
    output_image_[i].resize(img_size_ - 2, 0);
  }
  return true;
}

bool ersoz_b_test_task_seq::TestTaskSequential::ValidationImpl() {
  unsigned int input_size = task_data->inputs_count[0];
  int computed_size = static_cast<int>(std::sqrt(input_size));
  if (static_cast<unsigned int>(computed_size * computed_size) != input_size) {
    return false;
  }
  if (task_data->outputs_count[0] != static_cast<unsigned int>((computed_size - 2) * (computed_size - 2))) {
    return false;
  }
  img_size_ = computed_size;
  return true;
}

bool ersoz_b_test_task_seq::TestTaskSequential::RunImpl() {
  output_image_ = SequentialGaussianFilter(input_image_, sigma_);
  return true;
}

bool ersoz_b_test_task_seq::TestTaskSequential::PostProcessingImpl() {
  uint8_t* out_ptr = task_data->outputs[0];
  int index = 0;
  for (const auto& row : output_image_) {
    for (char pixel : row) {
      out_ptr[index++] = static_cast<uint8_t>(pixel);
    }
  }
  return true;
}
