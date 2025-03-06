#pragma once

#include <boost/mpi/communicator.hpp>
#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

namespace somov_i_num_of_alternations_signs_mpi {
class NumOfAlternationsSigns : public ppc::core::Task {
 public:
  explicit NumOfAlternationsSigns(ppc::core::TaskDataPtr task_data) : Task(std::move(task_data)) {}
  bool PreProcessingImpl() override;
  bool ValidationImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

 private:
  boost::mpi::communicator world_;
  std::vector<int> input_;
  int sz_ = 0;
  int output_ = 0;
};

void CheckForAlternationSigns(const std::vector<int>& vec, int& out);
}  // namespace somov_i_num_of_alternations_signs_mpi