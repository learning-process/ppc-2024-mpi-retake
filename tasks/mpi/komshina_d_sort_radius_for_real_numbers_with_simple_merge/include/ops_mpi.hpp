#pragma once

#include <mpi.h>

#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

namespace komshina_d_sort_radius_for_real_numbers_with_simple_merge_mpi {

class TestTaskMPI : public ppc::core::Task {
 public:
  explicit TestTaskMPI(ppc::core::TaskDataPtr task_data) : Task(std::move(task_data)) {
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank_);
    MPI_Comm_size(MPI_COMM_WORLD, &size_);
  }
  bool PreProcessingImpl() override;
  bool ValidationImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

 private:
  void ExecuteParallelSorting();
  static void ApplyRadixSorting(std::vector<double>& data);
  static void ProcessAndSortSignedNumbers(std::vector<double>& data);

  int world_rank_;
  int size_;
  std::vector<double> input_data_;
  std::vector<double> sortedData_;
};

}  // namespace komshina_d_sort_radius_for_real_numbers_with_simple_merge_mpi