#include "seq/Konstantinov_I_histogram_stretching/include/ops_seq.hpp"

#include <random>
#include <thread>

std::vector<int> konstantinov_i_linear_histogram_stretch_seq::GetRandomImage(int sz) {
  std::mt19937 gen(42);
  std::uniform_int_distribution<int> dist(0, 255);
  std::vector<int> vec(sz);
  for (int i = 0; i < sz; i++) {
    vec[i] = dist(gen);
  }
  return vec;
}

bool konstantinov_i_linear_histogram_stretch_seq::LinearHistogramStretchSeq::PreProcessingImpl() {
  int size = static_cast<int>(task_data->inputs_count[0]);
  image_input_ = std::vector<int>(size);
  auto* tmp_ptr = reinterpret_cast<int*>(task_data->inputs[0]);
  std::copy(tmp_ptr, tmp_ptr + size, image_input_.begin());

  int pixel_count = size / 3;
  I_.resize(pixel_count);
  for (int i = 0, k = 0; i < size; i += 3, ++k) {
    int r = image_input_[i];
    int g = image_input_[i + 1];
    int b = image_input_[i + 2];

    I_[k] = static_cast<int>((0.299 * static_cast<double>(r)) + (0.587 * static_cast<double>(g)) +
                             (0.114 * static_cast<double>(b)));
  }

  image_output_ = {};
  return true;
}

bool konstantinov_i_linear_histogram_stretch_seq::LinearHistogramStretchSeq::ValidationImpl() {
  int size = static_cast<int>(task_data->inputs_count[0]);
  if (size % 3 != 0) {
    return false;
  }
  for (int i = 0; i < size; ++i) {
    int value = reinterpret_cast<int*>(task_data->inputs[0])[i];
    if (value < 0 || value > 255) {
      return false;
    }
  }

  return ((!task_data->inputs.empty() && !task_data->outputs.empty()) &&
          (!task_data->inputs_count.empty() && task_data->inputs_count[0] != 0) &&
          (!task_data->outputs_count.empty() && task_data->outputs_count[0] != 0));
}

bool konstantinov_i_linear_histogram_stretch_seq::LinearHistogramStretchSeq::RunImpl() {
  int size = static_cast<int>(image_input_.size());
  image_output_.resize(size);
  int imin = 255;
  int imax = 0;

  for (int intensity : I_) {
    imin = std::min(imin, intensity);
    imax = std::max(imax, intensity);
  }

  if (imin == imax) {
    image_output_ = image_input_;
    return true;
  }

  for (int i = 0, k = 0; i < size; i += 3, ++k) {
    int inew = ((I_[k] - imin) * 255) / (imax - imin);

    float coeff = static_cast<float>(inew) / static_cast<float>(I_[k]);

    image_output_[i] = std::min(255, static_cast<int>(image_input_[i] * coeff));
    image_output_[i + 1] = std::min(255, static_cast<int>(image_input_[i + 1] * coeff));
    image_output_[i + 2] = std::min(255, static_cast<int>(image_input_[i + 2] * coeff));
  }

  return true;
}

bool konstantinov_i_linear_histogram_stretch_seq::LinearHistogramStretchSeq::PostProcessingImpl() {
  auto* output = reinterpret_cast<int*>(task_data->outputs[0]);
  std::ranges::copy(image_output_, output);
  return true;
}