#pragma once

<<<<<<< HEAD
#include <mpi.h>

#include <numeric>  // Для std::accumulate
=======
#include <boost/mpi/collectives.hpp>
#include <boost/mpi/communicator.hpp>
>>>>>>> 8c0b0a4bb0e393c52cb48d47e5dccf68736a6c16
#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

namespace karaseva_e_reduce_mpi {

<<<<<<< HEAD
template <typename T>
class TestTaskMPI : public ppc::core::Task {
 public:
  explicit TestTaskMPI(ppc::core::TaskDataPtr task_data) : Task(std::move(task_data)) {}
=======
class TestTaskMPI : public ppc::core::Task {
 public:
  explicit TestTaskMPI(ppc::core::TaskDataPtr task_data, int size) : Task(std::move(task_data)), size_(size) {
    input_.resize(size_);
    output_.resize(size_);
  }

>>>>>>> 8c0b0a4bb0e393c52cb48d47e5dccf68736a6c16
  bool PreProcessingImpl() override;
  bool ValidationImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

 private:
<<<<<<< HEAD
  std::vector<T> input_, output_;
  int rc_size_{};  // Размер данных

  void ReduceBinaryTree(T* local_data, T& global_data, int root);
=======
  std::vector<int> input_, output_;
  int size_;

  void ReduceBinaryTree(boost::mpi::communicator& world);
>>>>>>> 8c0b0a4bb0e393c52cb48d47e5dccf68736a6c16
};

}  // namespace karaseva_e_reduce_mpi