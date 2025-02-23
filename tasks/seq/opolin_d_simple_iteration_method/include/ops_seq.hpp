// Copyright 2024 Nesterov Alexander
#pragma once

#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

namespace opolin_d_simple_iteration_method_seq {

size_t Rank(std::vector<double> matrix, size_t n);
bool IsDiagonalDominance(std::vector<double> mat, size_t dim);

class TestTaskSequential : public ppc::core::Task {
 public:
  explicit TestTaskSequential(ppc::core::TaskDataPtr task_data) : Task(std::move(task_data)) {}
  bool PreProcessingImpl() override;
  bool ValidationImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

 private:
  std::vector<double> A_;
  std::vector<double> C_;
  std::vector<double> b_;
  std::vector<double> d_;
  std::vector<double> Xold_;
  std::vector<double> Xnew_;
  double epsilon_;
  uint32_t n_;
  int max_iter_;
};

}  // namespace opolin_d_simple_iteration_method_seq