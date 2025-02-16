#pragma once

#include <boost/mpi/collectives.hpp>
#include <boost/mpi/communicator.hpp>
#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

namespace karaseva_e_reduce_mpi {

class TestTaskMPI : public ppc::core::Task {
 public:
 explicit TestTaskMPI(ppc::core::TaskDataPtr task_data, int size) : Task(std::move(task_data)), size_(size) {
    input_.resize(size_);
    output_.resize(size_);
  }

  bool PreProcessingImpl() override;
  bool ValidationImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

 private:
  std::vector<int> input_, output_;
  int size_;
  int rc_size_{};  // aftter reduce

  void ReduceBinaryTree(boost::mpi::communicator& world);
};

}  // namespace karaseva_e_reduce_mpi