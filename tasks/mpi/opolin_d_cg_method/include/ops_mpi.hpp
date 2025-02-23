// Copyright 2024 Nesterov Alexander
#pragma once

#include <boost/mpi/collectives.hpp>
#include <boost/mpi/communicator.hpp>
#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

namespace opolin_d_cg_method_mpi {

bool isPositiveDefinite(const std::vector<double>& mat, size_t size);
bool isSimmetric(const std::vector<double>& mat, size_t size);
double scalarProduct(const std::vector<double>& a_, const std::vector<double>& b_);
std::vector<double> multiplyVecMat(const std::vector<double>& vec, const std::vector<double>& mat);

class CGMethodkMPI : public ppc::core::Task {
 public:
  explicit CGMethodkMPI(ppc::core::TaskDataPtr task_data) : Task(std::move(task_data)) {}
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

}  // namespace opolin_d_cg_method_mpi