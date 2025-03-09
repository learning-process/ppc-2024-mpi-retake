// Copyright 2023 Nesterov Alexander
#pragma once

#include <boost/mpi/collectives.hpp>
#include <boost/mpi/communicator.hpp>
#include <memory>
#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

namespace shishkarev_a_dijkstra_algorithm_mpi {

struct Matrix {
  std::vector<int> values;
  std::vector<int> col_index;
  std::vector<int> row_ptr;
};

void ConvertToCrs(const std::vector<int>& w, Matrix& matrix, int n);

class TestMPITaskSequential : public ppc::core::Task {
 public:
  explicit TestMPITaskSequential(std::shared_ptr<ppc::core::TaskData> task_data) : Task(std::move(task_data)) {}
  bool PreProcessingImpl() override;
  bool ValidationImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

 private:
  std::vector<int> input_;
  std::vector<int> res_;
  int st_{};
  int size_{};
};

class TestMPITaskParallel : public ppc::core::Task {
 public:
  explicit TestMPITaskParallel(std::shared_ptr<ppc::core::TaskData> task_data) : Task(std::move(task_data)) {}
  bool PreProcessingImpl() override;
  bool ValidationImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

 private:
  std::vector<int> input_;
  std::vector<int> res_;
  std::vector<int> values_;
  std::vector<int> col_index_;
  std::vector<int> row_ptr_;
  int st_{};
  int size_{};
  boost::mpi::communicator world_;
};

}  // namespace shishkarev_a_dijkstra_algorithm_mpi