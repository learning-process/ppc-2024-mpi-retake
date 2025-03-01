#pragma once

#include <boost/mpi/communicator.hpp>

#include "core/task/include/task.hpp"

namespace konkov_i_task_dining_philosophers_mpi {

class DiningPhilosophersMPI : public ppc::core::Task {
 public:
  explicit DiningPhilosophersMPI(ppc::core::TaskDataPtr task_data) : Task(std::move(task_data)), world_() {
    if (!this->task_data->inputs_count.empty()) {
      num_philosophers_ = this->task_data->inputs_count[0];
    }
  }

  bool PreProcessingImpl() override;
  bool ValidationImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

 private:
  int num_philosophers_{};
  boost::mpi::communicator world_;
};

}  // namespace konkov_i_task_dining_philosophers_mpi
