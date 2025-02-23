#pragma once

#include <gtest/gtest.h>

#include <memory>
#include <numeric>
#include <string>
#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

namespace shishkarev_a_gaussian_method_horizontal_strip_pattern_seq {

int MatrixRank(int n, int m, std::vector<double> a);

int Determinant(int n, int m, std::vector<double> a);

template <class InOutType>
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

}  // namespace shishkarev_a_gaussian_method_horizontal_strip_pattern_seq