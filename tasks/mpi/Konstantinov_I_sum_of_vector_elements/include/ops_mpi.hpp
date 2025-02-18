#pragma once

#include <boost/mpi/collectives.hpp>
#include <boost/mpi/communicator.hpp>
#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

namespace Konstantinov_I_sum_of_vector_elements_mpi {

int vec_elem_sum(const std::vector<int>& vec);

class SumVecElemSequential : public ppc::core::Task {
 public:
  explicit SumVecElemSequential(std::shared_ptr<ppc::core::TaskData> taskData) : Task(std::move(taskData)) {}
  bool PreProcessingImpl() override;
  bool ValidationImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

 private:
  std::vector<int> input_;
  int result_{};
};

class SumVecElemParallel : public ppc::core::Task {
 public:
  explicit SumVecElemParallel(ppc::core::TaskDataPtr task_data) : Task(std::move(task_data)) {}
  bool PreProcessingImpl() override;
  bool ValidationImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

 private:
  std::vector<int> input_, output_;
  int result_{};
  boost::mpi::communicator world_;
};

}  // namespace Konstantinov_I_sum_of_vector_elements_mpi