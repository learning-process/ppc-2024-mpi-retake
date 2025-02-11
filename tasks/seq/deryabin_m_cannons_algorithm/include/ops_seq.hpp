#pragma once

#include <cmath>
#include <numeric>
#include <vector>

#include "core/task/include/task.hpp"

namespace deryabin_m_cannons_algorithm_seq {

class CannonsAlgorithmTaskSequential : public ppc::core::Task {
 public:
  explicit CannonsAlgorithmTaskSequential(ppc::core::TaskDataPtr task_data)
      : Task(std::move(task_data)) {}
  bool PreProcessingImpl() override;
  bool ValidationImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

 private:
  std::vector<double> input_matrix_A;
  std::vector<double> input_matrix_B;
  std::vector<double> output_matrix_C;
};

}  // namespace deryabin_m_cannons_algorithm_seq
