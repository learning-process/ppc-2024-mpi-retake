#pragma once

#include <boost/mpi/collectives.hpp>
#include <boost/mpi/communicator.hpp>
#include <cmath>
#include <functional>
#include <memory>
#include <utility>

#include "core/task/include/task.hpp"

namespace kabalova_v_strongin_mpi {

class TestMPITaskSequential : public ppc::core::Task {
 public:
  explicit TestMPITaskSequential(std::shared_ptr<ppc::core::TaskData> task_data, std::function<double(double *)> f)
      : Task(std::move(task_data)), f_(std::move(f)) {}
  bool PreProcessingImpl() override;
  bool ValidationImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

 private:
  double left_{};
  double right_{};
  std::function<double(double *)> f_;
  std::pair<double, double> result_;
};
double Algorithm(double left, double right, const std::function<double(double *)> &f, double eps = 0.0001);
std::pair<double, double> GenerateBounds(double left = -5.0, double right = 5.0);

class TestMPITaskParallel : public ppc::core::Task {
 public:
  explicit TestMPITaskParallel(std::shared_ptr<ppc::core::TaskData> task_data, std::function<double(double *)> f)
      : Task(std::move(task_data)), f_(std::move(f)) {}
  bool PreProcessingImpl() override;
  bool ValidationImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

 private:
  double left_{};
  double right_{};
  std::function<double(double *)> f_;
  std::pair<double, double> result_;
  boost::mpi::communicator world_;
};

}  // namespace kabalova_v_strongin_mpi