#pragma once

#include <boost/mpi/collectives.hpp>
#include <boost/mpi/communicator.hpp>
#include <utility>
#include <algorithm>
#include <vector>
#include <memory>

#include "core/task/include/task.hpp"

namespace komshina_d_sort_radius_for_real_numbers_with_simple_merge_mpi {

class TestTaskMPI : public ppc::core::Task {
 public:
  explicit TestTaskMPI(ppc::core::TaskDataPtr task_data) : Task(std::move(task_data)) {
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank_);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
  }
  bool PreProcessingImpl() override;
  bool ValidationImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

 private:
  void ExecuteParallelSorting();
  void ApplyRadixSorting(std::vector<double>& data);
  void ProcessAndSortSignedNumbers(std::vector<double>& data);

  int world_rank_;
  int size;
  std::vector<double> input_data_;
  std::vector<double> sortedData;
};

}  // namespace komshina_d_sort_radius_for_real_numbers_with_simple_merge_mpi