#pragma once

#include <boost/mpi/collectives.hpp>
#include <boost/mpi/communicator.hpp>
#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

namespace karaseva_e_num_of_alternations_signs_mpi {

class AlternatingSignsMPI : public ppc::core::Task {
 public:
  explicit AlternatingSignsMPI(ppc::core::TaskDataPtr task_data) : Task(std::move(task_data)) {}
  bool PreProcessingImpl() override;
  bool ValidationImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

 private:
  std::vector<int> input_, output_;
  int total_{};
  boost::mpi::communicator world_;
};

}  // namespace karaseva_e_num_of_alternations_signs_mpi