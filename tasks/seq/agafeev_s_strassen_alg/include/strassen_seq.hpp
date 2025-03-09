#pragma once

#include <limits>
#include <random>
#include <string>
#include <vector>
#include <ctime>

#include "core/task/include/task.hpp"

namespace agafeev_s_strassen_alg_seq {

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

std::vector<double> mergeMatrices(const std::vector<double>& A11, const std::vector<double>& A12, const std::vector<double>& A21, const std::vector<double>& A22, int n);
std::vector<double> addMatrices(const std::vector<double>& A, const std::vector<double>& B, int n);
std::vector<double> subtractMatrices(const std::vector<double>& A, const std::vector<double>& B, int n);
std::vector<double> strassenMultiply(const std::vector<double>& A, const std::vector<double>& B, int n);
void splitMatrix(const std::vector<double>& A, std::vector<double>& A11, std::vector<double>& A12, std::vector<double>& A21, std::vector<double>& A22, int n);

}  // namespace agafeev_s_strassen_alg_seq