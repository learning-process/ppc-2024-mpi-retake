#pragma once

#include <gtest/gtest.h>

#include <boost/mpi/collectives.hpp>
#include <boost/mpi/communicator.hpp>
#include <memory>
#include <numeric>
#include <string>
#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

namespace Shpynov_N_readers_writers_mpi {
class TestTaskMPI : public ppc::core::Task {
public:
  explicit TestTaskMPI(ppc::core::TaskDataPtr task_data)
      : Task(std::move(task_data)) {}
  bool PreProcessingImpl() override;
  bool ValidationImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

private:
  std::vector<int> critical_resource;
  std::vector<int> result;
  boost::mpi::communicator world;
};
} // namespace Shpynov_N_readers_writers_mpi