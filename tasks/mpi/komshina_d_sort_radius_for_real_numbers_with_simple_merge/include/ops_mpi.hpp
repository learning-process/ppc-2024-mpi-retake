#pragma once

#include <mpi.h>

#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

namespace komshina_d_sort_radius_for_real_numbers_with_simple_merge_mpi {

class TestTaskMPI : public ppc::core::Task {
 public:
  explicit TestTaskMPI(ppc::core::TaskDataPtr task_data) : Task(std::move(task_data)) {
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
  }

  bool PreProcessingImpl() override;
  bool ValidationImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

 private:
  int rank, size;
  std::vector<double> input_data_, sorted_data_;
  void parallelSort();
  static void radixSort(std::vector<double>& data);
  static void handleSignAndSort(std::vector<double>& data);
};

}  // namespace komshina_d_sort_radius_for_real_numbers_with_simple_merge_mpi