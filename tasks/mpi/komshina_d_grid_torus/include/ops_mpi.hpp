#pragma once

#include <boost/mpi/collectives.hpp>
#include <boost/mpi/communicator.hpp>
#include <memory>
#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

namespace komshina_d_grid_torus_topology_mpi {

class TestTaskMPI : public ppc::core::Task {
 public:
  explicit TestTaskMPI(std::shared_ptr<ppc::core::TaskData> task_data) : Task(std::move(task_data)) {}
  bool PreProcessingImpl() override;
  bool ValidationImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;
  static std::vector<int> ComputeNeighbors(int rank, int grid_size);

 private:
  boost::mpi::communicator world_;
  boost::mpi::status stat_;
  friend class TestNeighborOutOfBounds_Test;
};

}  // namespace komshina_d_grid_torus_topology_mpi