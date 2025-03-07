#pragma once
#include <memory>
#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

namespace konstantinov_i_gauss_jordan_method_seq {

std::vector<double> ProcessMatrix(int n, int k, const std::vector<double>& matrix);
void UpdateMatrix(int n, int k, std::vector<double>& matrix, const std::vector<double>& iter_result);

class GaussJordanMethodSeq : public ppc::core::Task {
 public:
  explicit GaussJordanMethodSeq(std::shared_ptr<ppc::core::TaskData> task_data) : Task(std::move(task_data)) {}
  bool PreProcessingImpl() override;
  bool ValidationImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

 private:
  std::vector<double> matrix_;
  int n_;
};

}  // namespace konstantinov_i_gauss_jordan_method_seq