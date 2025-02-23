#pragma once

#include <gtest/gtest.h>

#include <algorithm>
#include <boost/mpi/communicator.hpp>
#include <boost/mpi/environment.hpp>
#include <boost/serialization/vector.hpp>
#include <cstdint>
#include <cstring>
#include <memory>
#include <vector>

#include "core/task/include/task.hpp"

namespace kavtorev_d_radix_double_sort {

class RadixSortSequential : public ppc::core::Task {
 public:
  explicit RadixSortSequential(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}
  bool PreProcessingImpl() override;
  bool ValidationImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

 private:
  std::vector<double> data_;
  int n = 0;

  static void radix_sort_doubles(std::vector<double>& data_);
  static void radix_sort_uint64(std::vector<uint64_t>& keys_);
};

class RadixSortParallel : public ppc::core::Task {
 public:
  explicit RadixSortParallel(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}
  bool PreProcessingImpl() override;
  bool ValidationImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

 private:
  std::vector<double> data_;
  int n = 0;
  boost::mpi::communicator world;

  static void radix_sort_doubles(std::vector<double>& data_);
  static void radix_sort_uint64(std::vector<uint64_t>& keys_);
};

}  // namespace kavtorev_d_radix_double_sort