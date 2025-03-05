#pragma once

#include <boost/mpi/collectives.hpp>
#include <boost/mpi/communicator.hpp>
#include <functional>
#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

namespace prokhorov_n_global_search_algorithm_strongin_mpi {

class TestTaskMPI : public ppc::core::Task {
 public:
  explicit TestTaskMPI(ppc::core::TaskDataPtr task_data, std::function<double(double)> func)
      : Task(std::move(task_data)), f_(std::move(func)), world_() {}
  bool PreProcessingImpl() override;
  bool ValidationImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

 private:
  double a_{};
  double b_{};
  double epsilon_{};
  double result_{};

  std::function<double(double)> f_;
  boost::mpi::communicator world_;

  double StronginAlgorithm(double local_a, double local_b);
};

}  // namespace prokhorov_n_global_search_algorithm_strongin_mpi