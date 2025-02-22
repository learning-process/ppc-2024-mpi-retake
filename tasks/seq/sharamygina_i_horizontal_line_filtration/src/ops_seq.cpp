#include "seq/sharamygina_i_horizontal_line_filtration/include/ops_seq.h"

bool sharamygina_i_horizontal_line_filtration_seq::horizontal_line_filtration_seq::PreProcessingImpl() {
  rows_ = task_data->inputs_count[0];
  cols_ = task_data->inputs_count[1];

  auto* input_buffer = reinterpret_cast<unsigned int*>(task_data->inputs[0]);
  original_data_.assign(input_buffer, input_buffer + rows_ * cols_);
  result_data_.resize(rows_ * cols_, 0);

  return true;
}

bool sharamygina_i_horizontal_line_filtration_seq::horizontal_line_filtration_seq::ValidationImpl() {
  if (task_data->inputs_count[0] < 3 || task_data->inputs_count[1] < 3 || !task_data || task_data->inputs.empty() ||
      task_data->inputs_count.size() < 2 || task_data->outputs.empty() || task_data->outputs_count.empty()) {
    return false;
  }

  return true;
}

bool sharamygina_i_horizontal_line_filtration_seq::horizontal_line_filtration_seq::RunImpl() {
  for (int i = 0; i < rows_; i++) {
    for (int j = 0; j < cols_; j++) {
      result_data_[i * cols_ + j] = InputAnotherPixel(original_data_, i, j, rows_, cols_);
    }
  }

  return true;
}

bool sharamygina_i_horizontal_line_filtration_seq::horizontal_line_filtration_seq::PostProcessingImpl() {
  unsigned int* output_ptr = reinterpret_cast<unsigned int*>(task_data->outputs[0]);
  std::copy(result_data_.begin(), result_data_.end(), output_ptr);
  return true;
}

unsigned int sharamygina_i_horizontal_line_filtration_seq::horizontal_line_filtration_seq::InputAnotherPixel(
    const std::vector<unsigned int>& image, int x, int y, int rows, int cols) {
  if (x < 1 || x >= rows - 1 || y < 1 || y >= cols - 1) {
    return 0;  
  }
  unsigned int sum = 0;
  for (int i = 0; i < 3; i++)
    for (int j = 0; j < 3; j++) {
      int tX = x + i - 1, tY = y + j - 1;
      if (tX < 0 || tX > rows_ - 1) tX = x;
      if (tY < 0 || tY > cols_ - 1) tY = y;
      if (tX * cols + tY >= cols * rows) {
        tX = x;
        tY = y;
      }
      sum += static_cast<unsigned int>(image[tX * cols + tY] * (gauss[i][j]));
    }
  return sum / 16;
}
