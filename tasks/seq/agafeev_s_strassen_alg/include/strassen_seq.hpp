#pragma once

#include <limits>
#include <random>
#include <string>
#include <vector>

#include "core/task/include/task.hpp"

namespace agafeev_s_strassen_alg_seq {

template <typename T>
std::vector<T> matrix_Multiply(const std::vector<T>& A, const std::vector<T>& B, int rowColSize) {
  std::vector<int> C(rowColSize * rowColSize, 0);

  for (int i = 0; i < rowColSize; ++i) {
    for (int j = 0; j < rowColSize; ++j) {
      for (int k = 0; k < rowColSize; ++k) {
        C[i * rowColSize + j] += A[i * rowColSize + k] * B[k * rowColSize + j];
      }
    }
  }

  return C;
}

class MultiplMatrixSequental : public ppc::core::Task {
 public:
  explicit MultiplMatrixSequental(std::shared_ptr<ppc::core::TaskData> task_data) : Task(std::move(task_data)) {}
  bool PreProcessingImpl() override;
  bool ValidationImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

 private:
  std::vector<double> first_input_;
  std::vector<double> second_input_;
  std::vector<double> result_;
};

}  // namespace agafeev_s_strassen_alg_seq