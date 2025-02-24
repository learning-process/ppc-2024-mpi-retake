// Copyright 2023 Nesterov Alexander
#pragma once

#include <boost/mpi/collectives.hpp>
#include <boost/mpi/communicator.hpp>
#include <cstdint>
#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

namespace makhov_m_ring_topology {

class TestMPITaskParallel : public ppc::core::Task {
 public:
  explicit TestMPITaskParallel(ppc::core::TaskDataPtr task_data) : Task(std::move(task_data)) {}
  bool PreProcessingImpl() override;
  bool ValidationImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

 private:
  std::vector<int32_t> input_data_;
  std::vector<int32_t> output_data_;
  std::vector<int32_t> sequence_;
  boost::mpi::communicator world_;
};

}  // namespace makhov_m_ring_topology