#pragma once

#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

namespace shkurinskaya_e_fox_mat_mul_seq {
std::vector<double> GetRandomMatrix(int rows, int cols);
class FoxMatMulSequential : public ppc::core::Task {
 public:
  explicit FoxMatMulSequential(ppc::core::TaskDataPtr task_data) : Task(std::move(task_data)) {}
  bool PreProcessingImpl() override;
  bool ValidationImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

 private:
  std::vector<double> inputA_, inputB_, output_;
  int matrix_size_;
};

}  // namespace shkurinskaya_e_fox_mat_mul_seq