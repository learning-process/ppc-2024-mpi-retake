#pragma once

#include <boost/mpi/collectives.hpp>
#include <boost/mpi/communicator.hpp>
#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

namespace khokhlov_a_sum_values_by_rows_mpi {
class SumValByRowsMpi : public ppc::core::Task {
 public:
  explicit SumValByRowsMpi(ppc::core::TaskDataPtr task_data) : Task(std::move(task_data)) {}
  bool PreProcessingImpl() override;
  bool ValidationImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

 private:
  std::vector<int> input_, local_input_;
  unsigned int row_, col_;
  std::vector<int> sum_;
  boost::mpi::communicator world_;
};

}  // namespace khokhlov_a_sum_values_by_rows_mpi