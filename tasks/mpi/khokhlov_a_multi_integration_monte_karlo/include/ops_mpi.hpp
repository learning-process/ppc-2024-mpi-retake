#pragma once

#include <boost/mpi/collectives.hpp>
#include <boost/mpi/communicator.hpp>
#include <functional>
#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

namespace khokhlov_a_multi_integration_monte_karlo_mpi {

class MonteCarloMpi : public ppc::core::Task {
 public:
  explicit MonteCarloMpi(ppc::core::TaskDataPtr task_data) : Task(std::move(task_data)) {}
  bool PreProcessingImpl() override;
  bool ValidationImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

  std::function<double(const std::vector<double>&)> integrand;

 private:
  boost::mpi::communicator world_;
  unsigned int dimension_;
  unsigned int N_;
  std::vector<double> lower_bound_, local_l_bound_;
  std::vector<double> upper_bound_, local_u_bound_;
  double result_;
};

}  // namespace khokhlov_a_multi_integration_monte_karlo_mpi