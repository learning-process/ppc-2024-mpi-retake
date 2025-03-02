#pragma once

#include <gtest/gtest.h>

#include <boost/mpi/collectives.hpp>
#include <boost/mpi/communicator.hpp>
#include <functional>
#include <memory>
#include <numeric>
#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

namespace kabalova_v_strongin_mpi {

class TestMPITaskSequential : public ppc::core::Task {
 public:
  explicit TestMPITaskSequential(std::shared_ptr<ppc::core::TaskData> taskData_, std::function<double(double*)> f_)
      : Task(std::move(taskData_)), f(f_) {}
  bool PreProcessingImpl() override;
  bool ValidationImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

 private:
  double left{};
  double right{};
  std::function<double(double*)> f;
  std::pair<double, double> result{};
};
double algorithm(const double a, const double b, std::function<double(double*)> func, const double eps = 0.0001);

class TestMPITaskParallel : public ppc::core::Task {
 public:
  explicit TestMPITaskParallel(std::shared_ptr<ppc::core::TaskData> taskData_, std::function<double(double*)> f_)
      : Task(std::move(taskData_)), f(f_) {}
  bool PreProcessingImpl() override;
  bool ValidationImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

 private:
  double left{};
  double right{};
  std::function<double(double*)> f;
  std::pair<double, double> result{};
  boost::mpi::communicator world;
};

}  // namespace kabalova_v_strongin_mpi