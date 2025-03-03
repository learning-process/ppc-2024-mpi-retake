#include "seq/sharamygina_i_horizontal_line_filtration/include/ops_seq.h"

#include <ranges>
#include <vector>

bool sharamygina_i_horizontal_line_filtration_seq::HorizontalLineFiltrationSeq::PreProcessingImpl() {
  rows_ = task_data->inputs_count[0];
  cols_ = task_data->inputs_count[1];

  auto* input_buffer = reinterpret_cast<unsigned int*>(task_data->inputs[0]);
  original_data_.assign(input_buffer, input_buffer + (rows_ * cols_));
  result_data_.resize(rows_ * cols_, 0);

  return true;
}

bool sharamygina_i_horizontal_line_filtration_seq::HorizontalLineFiltrationSeq::ValidationImpl() {
  return !(task_data->inputs_count[0] < 3 || task_data->inputs_count[1] < 3 || !task_data ||
           task_data->inputs.empty() || task_data->inputs_count.size() < 2 || task_data->outputs.empty() ||
           task_data->outputs_count.empty());
}

bool sharamygina_i_horizontal_line_filtration_seq::HorizontalLineFiltrationSeq::RunImpl() {
  for (int i = 0; i < rows_; i++) {
    for (int j = 0; j < cols_; j++) {
      result_data_[(i * cols_) + j] = InputAnotherPixel(original_data_, i, j, rows_, cols_);
    }
  }

  return true;
}

bool sharamygina_i_horizontal_line_filtration_seq::HorizontalLineFiltrationSeq::PostProcessingImpl() {
  auto* output_ptr = reinterpret_cast<unsigned int*>(task_data->outputs[0]);
  std::ranges::copy(result_data_, output_ptr);
  return true;
}

unsigned int sharamygina_i_horizontal_line_filtration_seq::HorizontalLineFiltrationSeq::InputAnotherPixel(
    const std::vector<unsigned int>& image, int x, int y, int rows, int cols) {
  if (x < 1 || x >= rows - 1 || y < 1 || y >= cols - 1) {
    return 0;
  }
  unsigned int sum = 0;
  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 3; j++) {
      int t_x = x + i - 1;
      int t_y = y + j - 1;
      if (t_x < 0 || t_x > rows_ - 1) {
        t_x = x;
      }
      if (t_y < 0 || t_y > cols_ - 1) {
        t_y = y;
      }
      if ((t_x * cols) + t_y >= cols * rows) {
        t_x = x;
        t_y = y;
      }
      sum += static_cast<unsigned int>(image[(t_x * cols) + t_y] * (gauss_[i][j]));
    }
  }
  return sum / 16;
}
