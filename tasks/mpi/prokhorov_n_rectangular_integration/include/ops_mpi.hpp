#pragma once

#include <boost/mpi/collectives.hpp>
#include <boost/mpi/communicator.hpp>
#include <functional>
#include <utility>

#include "core/task/include/task.hpp"

namespace prokhorov_n_rectangular_integration_mpi {

class TestTaskSequential : public ppc::core::Task {
 public:
  explicit TestTaskSequential(ppc::core::TaskDataPtr task_data) : Task(std::move(task_data)) {}

  bool PreProcessingImpl() override;
  bool ValidationImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

  void SetFunction(const std::function<double(double)>& func);

 private:
  static double Integrate(const std::function<double(double)>& f, double lower_bound, double upper_bound, int n);

  double lower_bound_{};
  double upper_bound_{};
  int n_{};
  double result_{};
  std::function<double(double)> f_;
};

class TestTaskMPI : public ppc::core::Task {
 public:
  explicit TestTaskMPI(ppc::core::TaskDataPtr task_data) : Task(std::move(task_data)) {}
  bool PreProcessingImpl() override;
  bool ValidationImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;
  void SetFunction(const std::function<double(double)>& func);

 private:
  double ParallelIntegrate(const std::function<double(double)>& f, double lower_bound, double upper_bound, int n);
  double lower_bound_{};
  double upper_bound_{};
  int n_{};
  double result_{};
  std::function<double(double)> f_;
  boost::mpi::communicator world_;
};

}  // namespace prokhorov_n_rectangular_integration_mpi