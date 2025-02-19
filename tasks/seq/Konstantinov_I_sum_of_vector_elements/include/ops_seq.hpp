#pragma once
#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

namespace Konstantinov_I_sum_of_vector_elements_seq {

int vec_elem_sum(const std::vector<int>& vec);
std::vector<int> generate_rand_vector(int size, int lower_bound = 0, int upper_bound = 50);
std::vector<std::vector<int>> generate_rand_matrix(int rows, int columns, int lower_bound = 0, int upper_bound = 50);

class SumVecElemSequential : public ppc::core::Task {
 public:
  explicit SumVecElemSequential(std::shared_ptr<ppc::core::TaskData> taskData) : Task(std::move(taskData)) {}
  bool PreProcessingImpl() override;
  bool ValidationImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

 private:
  std::vector<int> input_;
  int result_{};
};
}  // namespace Konstantinov_I_sum_of_vector_elements_seq