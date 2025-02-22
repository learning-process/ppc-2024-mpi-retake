#pragma once

#include <gtest/gtest.h>

#include <boost/mpi/collectives.hpp>
#include <boost/mpi/communicator.hpp>
#include <memory>
#include <numeric>
#include <string>
#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

namespace shishkarev_a_gaussian_method_horizontal_strip_pattern_mpi {

int matrix_rank(int n, int m, std::vector<double> a);

int determinant(int n, int m, std::vector<double> a);

class MPIGaussHorizontalSequential : public ppc::core::Task {
 public:
  explicit MPIGaussHorizontalSequential(std::shared_ptr<ppc::core::TaskData> task_data) : Task(std::move(task_data)) {}
  bool PreProcessingImpl() override;
  bool ValidationImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

 private:
  std::vector<double> matrix, res;
  int rows{}, cols{};
};

class MPIGaussHorizontalParallel : public ppc::core::Task {
 public:
  explicit MPIGaussHorizontalParallel(std::shared_ptr<ppc::core::TaskData> task_data) : Task(std::move(task_data)) {}
  bool PreProcessingImpl() override;
  bool ValidationImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

 private:
  std::vector<double> matrix, local_matrix, res, local_res;
  int rows{}, cols{};
  boost::mpi::communicator world;
};

}  // namespace shishkarev_a_gaussian_method_horizontal_strip_pattern_mpi