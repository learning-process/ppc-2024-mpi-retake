// Copyright 2023 Nesterov Alexander
#pragma once

#include <boost/mpi/collectives.hpp>
#include <boost/mpi/communicator.hpp>
#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

namespace leontev_n_average_mpi {

class MPIVecAvgParallel : public ppc::core::Task {
 public:
  explicit MPIVecAvgParallel(ppc::core::TaskDataPtr task_data) : Task(std::move(task_data)) {}
  bool PreProcessingImpl() override;
  bool ValidationImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

 private:
  std::vector<int> input_, local_input_;
  int res_{};
  boost::mpi::communicator world_;
};

}  // namespace leontev_n_average_mpi
