#pragma once

#include <boost/mpi/collectives.hpp>
#include <boost/mpi/communicator.hpp>
#include <memory>
#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

namespace shishkarev_a_gaussian_method_horizontal_strip_pattern_mpi {

struct Matrix {
  int rows;
  int cols;
};

int MatrixRank(Matrix matrix, std::vector<double> a);

double Determinant(Matrix matrix, std::vector<double> a);

std::vector<double> GetRandomMatrix(int sz);

double AxB(int n, int m, std::vector<double> a, std::vector<double> res);

void BroadcastMatrixSize();

std::vector<int> ComputeRowDistribution();

void DistributeMatrix(const std::vector<int>& row_num);

void ReceiveMatrix(int delta);

void ForwardElimination(std::vector<double>& row);

void BackSubstitution(std::vector<double>& row);

class MPIGaussHorizontalSequential : public ppc::core::Task {
 public:
  explicit MPIGaussHorizontalSequential(std::shared_ptr<ppc::core::TaskData> task_data) : Task(std::move(task_data)) {}
  bool PreProcessingImpl() override;
  bool ValidationImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

 private:
  std::vector<double> matrix_, res_;
  int rows_{}, cols_{};
};

class MPIGaussHorizontalParallel : public ppc::core::Task {
 public:
  explicit MPIGaussHorizontalParallel(std::shared_ptr<ppc::core::TaskData> task_data) : Task(std::move(task_data)) {}
  bool PreProcessingImpl() override;
  bool ValidationImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

 protected:
  std::vector<double> matrix_, local_matrix_, res_, local_res_;
  int delta_, rows_{}, cols_{};
  boost::mpi::communicator world_;
};

}  // namespace shishkarev_a_gaussian_method_horizontal_strip_pattern_mpi