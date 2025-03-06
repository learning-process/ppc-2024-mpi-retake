#pragma once

#include <boost/mpi/collectives.hpp>
#include <boost/mpi/communicator.hpp>
#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

namespace malyshev_v_lent_horizontal_mpi {

class MatVecMultMpi : public ppc::core::Task {
 public:
  explicit MatVecMultMpi(ppc::core::TaskDataPtr task_data) : Task(std::move(task_data)) {}
  bool PreProcessingImpl() override;
  bool ValidationImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

 private:
  std::vector<int> matrix_, vector_, local_matrix_, local_result_;
  unsigned int rows_{}, cols_{};
  boost::mpi::communicator world_;
};

}  // namespace malyshev_v_lent_horizontal_mpi