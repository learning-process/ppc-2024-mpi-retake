// Copyright 2024 Nesterov Alexander
#pragma once

#include <boost/mpi/collectives.hpp>
#include <boost/mpi/communicator.hpp>
#include <cstddef>
#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

namespace malyshev_v_conjugate_gradient_mpi {

bool IsPositiveDefinite(const std::vector<double>& mat, size_t size);
bool IsSymmetric(const std::vector<double>& mat, size_t size);
double ScalarProduct(const std::vector<double>& a, const std::vector<double>& b);

class ConjugateGradientMethod : public ppc::core::Task {
 public:
  explicit ConjugateGradientMethod(ppc::core::TaskDataPtr task_data) : Task(std::move(task_data)) {}
  bool PreProcessingImpl() override;
  bool ValidationImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

 private:
  std::vector<double> A_;
  std::vector<double> b_;
  std::vector<double> x_;
  size_t n_;
  double epsilon_;
  boost::mpi::communicator world_;
};

}  // namespace malyshev_v_conjugate_gradient_mpi