#pragma once

#include <gtest/gtest.h>

#include <boost/mpi/collectives.hpp>
#include <boost/mpi/communicator.hpp>
#include <random>
#include <vector>

#include "core/task/include/task.hpp"

namespace khokhlov_a_sum_values_by_rows_mpi {

std::vector<int> getRandomMatrix(int size);

class Sum_val_by_rows_seq : public ppc::core::Task {
 public:
  explicit Sum_val_by_rows_seq(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  std::vector<int> input_;
  int row, col;
  std::vector<int> sum;
};

class Sum_val_by_rows_mpi : public ppc::core::Task {
 public:
  explicit Sum_val_by_rows_mpi(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  std::vector<int> input_, local_input_;
  int row, col;
  std::vector<int> sum;
  boost::mpi::communicator world;
};

}  // namespace khokhlov_a_sum_values_by_rows_mpi