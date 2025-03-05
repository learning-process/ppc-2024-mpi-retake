#pragma once

#include <cmath>
#include <cstddef>
#include <vector>

#include "core/task/include/task.hpp"

namespace shpynov_n_radix_sort_seq {

inline int GetMaxAmountOfDigits(std::vector<int>& vec) {
  int Max = 0;
  for (int i = 0; i < vec.size(); i++) {
    if (std::abs(vec[i]) > Max) {
      Max = std::abs(vec[i]);
    }
  }
  int count = 0;
  while (Max) {
    Max /= 10;
    count++;
  }
  return count;
}

inline std::vector<int> SortBySingleDigit(std::vector<int>& vec, int DigitPlace) {
  std::vector<int> output(vec.size());
  std::vector<int> count(10, 0);
  for (size_t i = 0; i < vec.size(); i++) {
    count[(vec[i] / (int)std::pow(10, DigitPlace)) % 10]++;
  }
  for (int i = 1; i < 10; i++) {
    count[i] += count[i - 1];
  }
  for (int i = vec.size() - 1; i >= 0; i--) {
    output[count[(vec[i] / (int)std::pow(10, DigitPlace)) % 10] - 1] = vec[i];
    count[(vec[i] / (int)std::pow(10, DigitPlace)) % 10]--;
  }
  for (int i = 0; i < vec.size(); i++) {
    vec[i] = output[i];
  }
  return output;
}

inline std::vector<int> RadixSort(std::vector<int>& vec) {
  std::vector<int> vecPos;
  std::vector<int> vecNeg;
  int MaxPosNum = 0;
  int MaxNegNum = 0;
  for (int i = 0; i < vec.size(); i++) {
    if (vec[i] < 0) {
      vecNeg.push_back(std::abs(vec[i]));
      MaxNegNum = GetMaxAmountOfDigits(vecNeg);
    } else {
      vecPos.push_back(vec[i]);
      MaxPosNum = GetMaxAmountOfDigits(vecPos);
    }
  }
  for (int i = 0; i < MaxNegNum; i++) {
    SortBySingleDigit(vecNeg, i);
  }
  std::reverse(vecNeg.begin(), vecNeg.end());
  for (int i = 0; i < vecNeg.size(); i++) {
    vecNeg[i] = -vecNeg[i];
  }
  for (int i = 0; i < MaxPosNum; i++) {
    SortBySingleDigit(vecPos, i);
  }
  vec.clear();
  vec.insert(vec.end(), vecNeg.begin(), vecNeg.end());
  vec.insert(vec.end(), vecPos.begin(), vecPos.end());
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