#pragma once

#include <boost/mpi/collectives.hpp>
#include <boost/mpi/communicator.hpp>
#include <memory>
#include <numeric>
#include <string>
#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

namespace konstantinov_i_gauss_jordan_method_mpi {

class GaussJordanMethodSeq : public ppc::core::Task {
 public:
  explicit GaussJordanMethodSeq(std::shared_ptr<ppc::core::TaskData> task_data) : Task(std::move(task_data)) {}
  bool PreProcessingImpl() override;
  bool ValidationImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

 private:
  int n = 0;
  std::vector<double> matrix;
  std::vector<double> solution;
};

class GaussJordanMethodMpi : public ppc::core::Task {
 public:
  explicit GaussJordanMethodMpi(std::shared_ptr<ppc::core::TaskData> task_data) : Task(std::move(task_data)) {}
  bool PreProcessingImpl() override;
  bool ValidationImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

 private:
  int n = 0;
  std::vector<double> matrix;
  std::vector<double> local_matrix;
  std::vector<double> solution;
  boost::mpi::communicator world;
  std::vector<double> diag_elements;
  std::vector<double> localMatrix;
  std::vector<double> header;
  std::vector<int> sendCounts;
  std::vector<int> displacements;
};
}  // namespace konstantinov_i_gauss_jordan_method_mpi