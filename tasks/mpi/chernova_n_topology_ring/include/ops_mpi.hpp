#pragma once

#include <boost/mpi/collectives.hpp>
#include <boost/mpi/communicator.hpp>
#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

namespace chernova_n_topology_ring_mpi {

class TestMPITaskParallel : public ppc::core::Task {
 public:
  explicit TestMPITaskParallel(ppc::core::TaskDataPtr task_data) : Task(std::move(task_data)) {}
  bool PreProcessingImpl() override;
  bool ValidationImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

 private:
  std::vector<char> input_;
  std::vector<int> process_;
  std::vector<char> output_;
  int vector_size_{};
  boost::mpi::communicator world_;
};

}  // namespace chernova_n_topology_ring_mpi 
