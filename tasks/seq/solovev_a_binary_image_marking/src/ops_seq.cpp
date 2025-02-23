#include <algorithm>
#include <iostream>
#include <map>
#include <queue>
#include <unordered_map>
#include <vector>

#include "seq/solovev_a_binary_image_marking/include/ops_sec.hpp"

bool solovev_a_binary_image_marking::TestTaskSequential::PreProcessingImpl() {
  int m_tmp = *reinterpret_cast<int *>(task_data->inputs[0]);
  int n_tmp = *reinterpret_cast<int *>(task_data->inputs[1]);

  std::vector<int> input_tmp;

  int *tmp_data = reinterpret_cast<int *>(task_data->inputs[2]);
  int tmp_size = task_data->inputs_count[2];
  input_tmp.assign(tmp_data, tmp_data + tmp_size);

  data.resize(tmp_size);
  labels.resize(tmp_size);

  data.assign(input_tmp.begin(), input_tmp.end());

  m = m_tmp;
  n = n_tmp;

  labels.assign(m * n, 0);

  return true;
}

bool solovev_a_binary_image_marking::TestTaskSequential::ValidationImpl() {
  int m_check = *reinterpret_cast<int *>(task_data->inputs[0]);
  int n_check = *reinterpret_cast<int *>(task_data->inputs[1]);

  std::vector<int> input_check;

  int *input_check_data = reinterpret_cast<int *>(task_data->inputs[2]);
  int input_check_size = task_data->inputs_count[2];
  input_check.assign(input_check_data, input_check_data + input_check_size);

  return (m_check > 0 && n_check > 0 && !input_check.empty());
}

bool solovev_a_binary_image_marking::TestTaskSequential::RunImpl() {
  std::vector<Point> directions = {{-1, 0}, {1, 0}, {0, -1}, {0, 1}};
  int label = 1;

  std::queue<Point> q;

  for (int i = 0; i < m; ++i) {
    for (int j = 0; j < n; ++j) {
      if (data[i * n + j] == 1 && labels[i * n + j] == 0) {
        q.push({i, j});
        labels[i * n + j] = label;

        while (!q.empty()) {
          Point current = q.front();
          q.pop();

          for (const Point &dir : directions) {
            int newX = current.x + dir.x;
            int newY = current.y + dir.y;

            if (newX >= 0 && newX < m && newY >= 0 && newY < n) {
              int newIdx = newX * n + newY;
              if (data[newIdx] == 1 && labels[newIdx] == 0) {
                labels[newIdx] = label;
                q.push({newX, newY});
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
  int *output = reinterpret_cast<int *>(task_data->outputs[0]);
  std::copy(labels.begin(), labels.end(), output);

  return true;
}