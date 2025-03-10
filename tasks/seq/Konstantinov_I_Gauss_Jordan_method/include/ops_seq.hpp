#pragma once
#include <memory>
#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

namespace konstantinov_i_gauss_jordan_method_seq {

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

}  // namespace konstantinov_i_gauss_jordan_method_seq