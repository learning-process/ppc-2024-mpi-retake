#include <algorithm>
#include <queue>
#include <ranges>
#include <vector>

#include "seq/solovev_a_binary_image_marking/include/ops_sec.hpp"

bool solovev_a_binary_image_marking::TestTaskSequential::PreProcessingImpl() {
  int m_tmp = *reinterpret_cast<int *>(task_data->inputs[0]);
  int n_tmp = *reinterpret_cast<int *>(task_data->inputs[1]);

  std::vector<int> input_tmp;

  int *tmp_data = reinterpret_cast<int *>(task_data->inputs[2]);
  int tmp_size = static_cast<int>(task_data->inputs_count[2]);
  input_tmp.assign(tmp_data, tmp_data + tmp_size);

  data_.resize(tmp_size);
  labels_.resize(tmp_size);

  data_.assign(input_tmp.begin(), input_tmp.end());

  m_ = m_tmp;
  n_ = n_tmp;

  labels_.assign(m_ * n_, 0);

  return true;
}

bool solovev_a_binary_image_marking::TestTaskSequential::ValidationImpl() {
  int m_check = *reinterpret_cast<int *>(task_data->inputs[0]);
  int n_check = *reinterpret_cast<int *>(task_data->inputs[1]);

  std::vector<int> input_check;

  int *input_check_data = reinterpret_cast<int *>(task_data->inputs[2]);
  int input_check_size = static_cast<int>(task_data->inputs_count[2]);
  input_check.assign(input_check_data, input_check_data + input_check_size);

  return (m_check > 0 && n_check > 0 && !input_check.empty());
}

bool solovev_a_binary_image_marking::TestTaskSequential::RunImpl() {
  std::vector<Point> directions = {{.x = -1, .y = 0}, {.x = 1, .y = 0}, {.x = 0, .y = -1}, {.x = 0, .y = 1}};
  int label = 1;

  std::queue<Point> q;

  for (int i = 0; i < m_; ++i) {
    for (int j = 0; j < n_; ++j) {
      if (data_[(i * n_) + j] == 1 && labels_[(i * n_) + j] == 0) {
        q.push({i, j});
        labels_[(i * n_) + j] = label;

        while (!q.empty()) {
          Point current = q.front();
          q.pop();

          for (const Point &dir : directions) {
            int new_x = current.x + dir.x;
            int new_y = current.y + dir.y;

            if (new_x >= 0 && new_x < m_ && new_y >= 0 && new_y < n_) {
              int new_idx = (new_x * n_) + new_y;
              if (data_[new_idx] == 1 && labels_[new_idx] == 0) {
                labels_[new_idx] = label;
                q.push({new_x, new_y});
              }
            }
          }
        }
        ++label;
      }
    }
  }

  return true;
}

bool solovev_a_binary_image_marking::TestTaskSequential::PostProcessingImpl() {
  int *output_ = reinterpret_cast<int *>(task_data->outputs[0]);
  std::ranges::copy(labels_, output_);

  return true;
}