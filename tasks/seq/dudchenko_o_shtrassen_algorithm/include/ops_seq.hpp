#pragma once

#include <cmath>
#include <cstddef>
#include <memory>
#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

namespace dudchenko_o_shtrassen_algorithm_seq {

class StrassenAlgoriphmSequential : public ppc::core::Task {
 public:
  explicit StrassenAlgoriphmSequential(std::shared_ptr<ppc::core::TaskData> task_data) : Task(std::move(task_data)) {}

  bool PreProcessingImpl() override;
  bool ValidationImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

 private:
  std::vector<double> matrixA_;
  std::vector<double> matrixB_;
  std::vector<double> result_;
  size_t size_;
};

std::vector<double> Add(const std::vector<double>& a, const std::vector<double>& b, size_t n);
std::vector<double> Subtract(const std::vector<double>& a, const std::vector<double>& b, size_t n);
std::vector<double> Strassen(const std::vector<double>& a, const std::vector<double>& b, size_t n);
}  // namespace dudchenko_o_shtrassen_algorithm_seq