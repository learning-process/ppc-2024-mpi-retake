// Copyright 2023 Nesterov Alexander
#pragma once

#include <gtest/gtest.h>

#include <boost/mpi/collectives.hpp>
#include <boost/mpi/communicator.hpp>
#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

namespace leontev_n_average_mpi {

class MPIVecAvgParallel : public ppc::core::Task {
 public:
  explicit MPIVecAvgParallel(std::shared_ptr<ppc::core::TaskData> task_data) : Task(std::move(task_data)) {}
  bool PreProcessingImpl() override;
  bool ValidationImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

 private:
  std::vector<int> input_, local_input_;
  int res{};
  boost::mpi::communicator world;
};

}  // namespace leontev_n_average_mpi
