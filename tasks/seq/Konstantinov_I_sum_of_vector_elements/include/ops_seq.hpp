#pragma once
#include <memory>
#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

namespace konstantinov_I_sum_of_vector_elements_seq {

int VecElemSum(const std::vector<int>& vec);
std::vector<int> GenerateRandVector(int size, int lower_bound = 0, int upper_bound = 50);
std::vector<std::vector<int>> GenerateRandMatrix(int rows, int columns, int lower_bound = 0, int upper_bound = 50);

class SumVecElemSequential : public ppc::core::Task {
 public:
  explicit SumVecElemSequential(std::shared_ptr<ppc::core::TaskData> task_data) : Task(std::move(task_data)) {}
  bool PreProcessingImpl() override;
  bool ValidationImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

 private:
  std::vector<int> input_;
  int result_{};
};
}  // namespace konstantinov_I_sum_of_vector_elements_seq