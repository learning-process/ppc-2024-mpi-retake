#pragma once

#include <boost/mpi/communicator.hpp>
#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

namespace ersoz_b_test_task_mpi {

class TestTaskMPI : public ppc::core::Task {
 public:
  explicit TestTaskMPI(ppc::core::TaskDataPtr task_data) : Task(std::move(task_data)) {}
  bool PreProcessingImpl() override;
  bool ValidationImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

 private:
  std::vector<std::vector<char>> input_image_;
  std::vector<std::vector<char>> output_image_;
  int img_size_{0};
  double sigma_{0.5};
  boost::mpi::communicator world_;
};

}  // namespace ersoz_b_test_task_mpi
