#pragma once

#include <boost/mpi/collectives.hpp>
#include <boost/mpi/communicator.hpp>
#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

namespace komshina_d_sort_radius_for_real_numbers_with_simple_merge_mpi {

class TestTaskMPI : public ppc::core::Task {
 public:
  explicit TestTaskMPI(ppc::core::TaskDataPtr task_data) : Task(std::move(task_data)) {}
  bool PreProcessingImpl() override;
  bool ValidationImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

 private:
  std::vector<double> numbers;
  int total_size = 0;

  void sort_doubles(std::vector<double>& arr);
  void sort_uint64(std::vector<uint64_t>& keys);
  boost::mpi::communicator world;
};

}  // namespace komshina_d_sort_radius_for_real_numbers_with_simple_merge_mpi