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
  void setInputData(const std::vector<double>& input);
  const std::vector<double>& getSortedData() const;
  static std::vector<double> generateRandomData(int size, double minValue = -10.0, double maxValue = 10.0);

 private:
  std::vector<double> input_data_;
  std::vector<double> sorted_data_;
  void parallelSort();

  int rank;
  int size;
};
void radixSortWithSignHandling(std::vector<double>& data);
void radixSort(std::vector<double>& data, int num_bits, int radix);

}  // namespace komshina_d_sort_radius_for_real_numbers_with_simple_merge_mpi