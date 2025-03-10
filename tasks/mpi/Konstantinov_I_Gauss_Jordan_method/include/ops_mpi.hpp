#pragma once

#include <boost/mpi/collectives.hpp>
#include <boost/mpi/communicator.hpp>
#include <cstddef>
#include <memory>
#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

namespace konstantinov_i_gauss_jordan_method_mpi {

void FindMaxRowAndSwap(int k, int n, std::vector<double>& matrix);
void NormalizeRow(int k, int n, std::vector<double>& matrix);
void ProcessLocalMatrix(size_t local_size, int k, int n, std::vector<double>& local_matrix,
                        const std::vector<double>& header);
void ProcessGaussStep(int k, int n, std::vector<double>& matrix, std::vector<double>& header,
                      std::vector<int>& send_counts, std::vector<int>& displacements, boost::mpi::communicator& world,
                      std::vector<double>& local_matrix, bool is_forward);

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