// Copyright 2023 Nesterov Alexander
#pragma once

#include <boost/mpi/collectives.hpp>
#include <boost/mpi/communicator.hpp>
#include <memory>
#include <numeric>
#include <string>
#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

namespace kalinin_d_vector_dot_product_mpi {
int vectorDotProduct(const std::vector<int>& v1, const std::vector<int>& v2);

class TestMPITaskSequential : public ppc::core::Task {
 public:
  explicit TestMPITaskSequential(ppc::core::TaskDataPtr task_data) : Task(std::move(task_data)) {}
  bool PreProcessingImpl() override;
  bool ValidationImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

 private:
  std::vector<std::vector<int>> input_;
  int res{};
};

class TestMPITaskParallel : public ppc::core::Task {
 public:
  explicit TestMPITaskParallel(ppc::core::TaskDataPtr task_data) : Task(std::move(task_data)) {}
  bool PreProcessingImpl() override;
  bool ValidationImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

 private:
  std::vector<std::vector<int>> input_{};
  std::vector<int> local_input1_{}, local_input2_{};
  std::vector<unsigned int> counts_{};
  size_t num_processes_ = 0;
  int res{};
  boost::mpi::communicator world;
};

}  // namespace kalinin_d_vector_dot_product_mpi