#pragma once

#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

namespace shkurinskaya_e_fox_mat_mul_seq {
std::vector<double> getRandomMatrix(int rows, int cols);
class FoxMatMulSequential : public ppc::core::Task {
 public:
  explicit FoxMatMulSequential(ppc::core::TaskDataPtr task_data) : Task(std::move(task_data)) {}
  bool PreProcessingImpl() override;
  bool ValidationImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

 private:
  std::vector<double> inputA, inputB, output;
  int matrix_size;
};

}  // namespace shkurinskaya_e_fox_mat_mul_seq