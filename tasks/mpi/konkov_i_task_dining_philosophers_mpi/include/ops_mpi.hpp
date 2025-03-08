#pragma once

#include <boost/mpi/collectives.hpp>
#include <boost/mpi/communicator.hpp>
#include <utility>

#include "core/task/include/task.hpp"

namespace konkov_i_task_dining_philosophers_mpi {

class DiningPhilosophersMPI : public ppc::core::Task {
 public:
  explicit DiningPhilosophersMPI(ppc::core::TaskDataPtr task_data) : Task(std::move(task_data)) {}
  bool PreProcessingImpl() override;
  bool ValidationImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

  bool DistributionForks();
  void ReleaseForks();
  bool CheckDeadlock();
  void ResolveDeadlock();
  bool CheckAllThink();

 private:
  boost::mpi::communicator world_;
  int status_;
  int l_philosopher_;
  int r_philosopher_;
  int count_philosophers_;
};

}  // namespace konkov_i_task_dining_philosophers_mpi