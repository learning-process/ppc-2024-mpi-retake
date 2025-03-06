#pragma once

#include <boost/mpi/collectives.hpp>
#include <boost/mpi/communicator.hpp>
#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

namespace dudchenko_o_sum_values_by_cols_mpi {
class SumValByColsMpi : public ppc::core::Task {
 public:
  explicit SumValByColsMpi(ppc::core::TaskDataPtr task_data) : Task(std::move(task_data)) {}
  bool PreProcessingImpl() override;
  bool ValidationImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

 private:
  std::vector<int> input_, local_input_;
  unsigned int rows_, cols_;
  std::vector<int> sum_;
  boost::mpi::communicator world_;
};

}  // namespace dudchenko_o_sum_values_by_cols_mpi