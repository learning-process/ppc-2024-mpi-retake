#pragma once

#include <boost/mpi/collectives.hpp>
#include <boost/mpi/communicator.hpp>
#include <boost/serialization/vector.hpp>  //NOLINT
#include <ctime>
#include <vector>
#include <memory> 
#include <utility>

#include "core/task/include/task.hpp"

namespace agafeev_s_strassen_alg_mpi {

class MultiplMatrixSequental : public ppc::core::Task {
 public:
  explicit MultiplMatrixSequental(std::shared_ptr<ppc::core::TaskData> task_data) : Task(std::move(task_data)) {}
  bool PreProcessingImpl() override;
  bool ValidationImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

 private:
  std::vector<double> first_input_;
  std::vector<double> second_input_;
  std::vector<double> result_;
  int size_;
};

class MultiplMatrixMpi : public ppc::core::Task {
 public:
  explicit MultiplMatrixMpi(std::shared_ptr<ppc::core::TaskData> task_data) : Task(std::move(task_data)) {}
  bool PreProcessingImpl() override;
  bool ValidationImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

  std::vector<double> StrassenMultiplyMpi(const std::vector<double>& a, const std::vector<double>& b, int n);

 private:
  std::vector<double> first_input_;
  std::vector<double> second_input_;
  std::vector<double> result_;
  int size_;
  boost::mpi::communicator world_;
};

std::vector<double> MergeMatrices(const std::vector<double>& a11, const std::vector<double>& a12,
                                  const std::vector<double>& a21, const std::vector<double>& a22, int n);
std::vector<double> AddMatrices(const std::vector<double>& a, const std::vector<double>& b, int n);
std::vector<double> SubtractMatrices(const std::vector<double>& a, const std::vector<double>& b, int n);
std::vector<double> StrassenMultiplySeq(const std::vector<double>& a, const std::vector<double>& b, int n);
void SplitMatrix(const std::vector<double>& a, std::vector<double>& a11, std::vector<double>& a12,
                 std::vector<double>& a21, std::vector<double>& a22, int n);

}  // namespace agafeev_s_strassen_alg_mpi