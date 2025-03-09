#pragma once
#include <boost/mpi.hpp>
#include <boost/mpi/collectives.hpp>
#include <boost/mpi/communicator.hpp>
#include <functional>
#include <limits>
#include <utility>
#include <vector>

#include "core/task/include/task.hpp"


namespace vasenkov_a_bellman_ford_mpi {

class BellmanFordMPI : public ppc::core::Task {
 public:
  explicit BellmanFordMPI(ppc::core::TaskDataPtr task_data) : Task(std::move(task_data)), world_() {}
  bool PreProcessingImpl() override;
  bool ValidationImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

 private:
  std::vector<int> row_ptr_;
  std::vector<int> col_ind_;
  std::vector<int> weights_;
  std::vector<int> distances_;
  int num_vertices_;
  int source_vertex_;
  boost::mpi::communicator world_;
};

class BellmanFordSequentialMPI : public ppc::core::Task {
 public:
  explicit BellmanFordSequentialMPI(ppc::core::TaskDataPtr task_data) : Task(std::move(task_data)) {}
  bool PreProcessingImpl() override;
  bool ValidationImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

 private:
  std::vector<int> row_ptr_;
  std::vector<int> col_ind_;
  std::vector<int> weights_;
  std::vector<int> distances_;
  int num_vertices_;
  int source_vertex_;
};

}  // namespace vasenkov_a_bellman_ford_mpi