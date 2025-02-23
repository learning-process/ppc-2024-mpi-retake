#pragma once

#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <memory>
#include <vector>

#include "core/task/include/task.hpp"

namespace kavtorev_d_radix_double_sort {

class RadixSortSequential : public ppc::core::Task {
 public:
  explicit RadixSortSequential(std::shared_ptr<ppc::core::TaskData> task_data) : Task(std::move(task_data)) {}
  bool PreProcessingImpl() override;
  bool ValidationImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

 private:
  std::vector<double> data_;
  int n = 0;

  static void radix_sort_doubles(std::vector<double>& data_);
  static void radix_sort_uint64(std::vector<uint64_t>& keys);
};

}  // namespace kavtorev_d_radix_double_sort