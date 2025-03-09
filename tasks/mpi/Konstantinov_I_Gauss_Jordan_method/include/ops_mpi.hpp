#pragma once

#include <boost/mpi/collectives.hpp>
#include <boost/mpi/communicator.hpp>
#include <memory>
#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

namespace konstantinov_i_gauss_jordan_method_mpi {

void FindMaxRowAndSwap(int k, int n_, std::vector<double>& matrix_);
void NormalizeRow(int k, int n_, std::vector<double>& matrix_);
void ProcessLocalMatrix(size_t local_size, int k, int n_, std::vector<double>& localMatrix_,
                        const std::vector<double>& header_);
void ProcessGaussStep(int k, int n_, std::vector<double>& matrix_, std::vector<double>& header_,
                      std::vector<int>& sendCounts_, std::vector<int>& displacements_, boost::mpi::communicator& world_,
                      std::vector<double>& localMatrix_, bool is_forward);

class GaussJordanMethodSeq : public ppc::core::Task {
 public:
  explicit GaussJordanMethodSeq(std::shared_ptr<ppc::core::TaskData> task_data) : Task(std::move(task_data)) {}
  bool PreProcessingImpl() override;
  bool ValidationImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

 private:
  int n_ = 0;
  std::vector<double> matrix_;
  std::vector<double> solution_;
};

class GaussJordanMethodMpi : public ppc::core::Task {
 public:
  explicit GaussJordanMethodMpi(std::shared_ptr<ppc::core::TaskData> task_data) : Task(std::move(task_data)) {}
  bool PreProcessingImpl() override;
  bool ValidationImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

 private:
  int n_ = 0;
  std::vector<double> matrix_;
  std::vector<double> local_matrix_;
  std::vector<double> solution_;
  boost::mpi::communicator world_;
  std::vector<double> diag_elements_;
  std::vector<double> localMatrix_;
  std::vector<double> header_;
  std::vector<int> sendCounts_;
  std::vector<int> displacements_;
};
}  // namespace konstantinov_i_gauss_jordan_method_mpi