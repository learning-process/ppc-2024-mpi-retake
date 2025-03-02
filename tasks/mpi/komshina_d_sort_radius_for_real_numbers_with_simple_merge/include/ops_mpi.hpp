#pragma once

#include <boost/mpi/collectives.hpp>
#include <boost/mpi/communicator.hpp>
#include <cstdint>
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
  std::vector<double> numbers_;
  int total_size_ = 0;

  static void SortDoubles(std::vector<double>& arr);
  static void SortUint64(std::vector<uint64_t>& keys);
  boost::mpi::communicator world_;
};

}  // namespace komshina_d_sort_radius_for_real_numbers_with_simple_merge_mpi