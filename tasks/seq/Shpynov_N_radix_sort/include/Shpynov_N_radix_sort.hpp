#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

namespace shpynov_n_radix_sort_seq {

inline int GetMaxAmountOfDigits(std::vector<int>& vec) {
  int maxNum = 0;
  for (int i = 0; i < (int)vec.size(); i++) {
    maxNum = std::max(std::abs(vec[i]), maxNum);
  }
  int count = 0;
  while (maxNum != 0) {
    maxNum /= 10;
    count++;
  }
  return count;
}

inline std::vector<int> SortBySingleDigit(std::vector<int>& vec, int digit_place) {
  std::vector<int> output(vec.size());
  std::vector<int> count(10, 0);
  for (int i = 0; i < (int)vec.size(); i++) {
    count[(vec[i] / (int)std::pow(10, digit_place)) % 10]++;
  }
  for (int i = 1; i < 10; i++) {
    count[i] += count[i - 1];
  }
  for (int i = (int)vec.size() - 1; i >= 0; i--) {
    output[count[(vec[i] / (int)std::pow(10, digit_place)) % 10] - 1] = vec[i];
    count[(vec[i] / (int)std::pow(10, digit_place)) % 10]--;
  }
  for (int i = 0; i < (int)vec.size(); i++) {
    vec[i] = output[i];
  }
  return output;
}

inline std::vector<int> RadixSort(std::vector<int>& vec) {
  std::vector<int> vec_pos;
  std::vector<int> vec_neg;
  int max_pos_num = 0;
  int max_neg_num = 0;
  for (int i = 0; i < (int)vec.size(); i++) {
    if (vec[i] < 0) {
      vec_neg.push_back(std::abs(vec[i]));
      max_neg_num = GetMaxAmountOfDigits(vec_neg);
    } else {
      vec_pos.push_back(vec[i]);
      max_pos_num = GetMaxAmountOfDigits(vec_pos);
    }
  }
  for (int i = 0; i < max_neg_num; i++) {
    SortBySingleDigit(vec_neg, i);
  }
  std::ranges::reverse(vec_neg.begin(), vec_neg.end());
  for (int i = 0; i < (int)vec_neg.size(); i++) {
    vec_neg[i] = -vec_neg[i];
  }
  for (int i = 0; i < max_pos_num; i++) {
    SortBySingleDigit(vec_pos, i);
  }
  vec.clear();
  vec.insert(vec.end(), vec_neg.begin(), vec_neg.end());
  vec.insert(vec.end(), vec_pos.begin(), vec_pos.end());
  return vec;
}

class TestTaskSEQ : public ppc::core::Task {
 public:
  explicit TestTaskSEQ(ppc::core::TaskDataPtr task_data) : Task(std::move(task_data)) {}
  bool PreProcessingImpl() override;
  bool ValidationImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

 private:
  std::vector<int> input_;
  std::vector<int> result_;
};

}  // namespace shpynov_n_radix_sort_seq