
#pragma once

#include <boost/mpi/collectives.hpp>
#include <boost/mpi/communicator.hpp>
#include <cstring>
#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

namespace solovev_a_binary_image_marking {

struct Point {
  int x, y;
};

class TestMPITaskSequential : public ppc::core::Task {
 public:
  explicit TestMPITaskSequential(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}
  bool PreProcessingImpl() override;
  bool ValidationImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

 private:
  std::vector<int> data_seq;
  std::vector<int> labels_seq;
  int m_seq, n_seq;
};

class TestMPITaskParallel : public ppc::core::Task {
 public:
  explicit TestMPITaskParallel(ppc::core::TaskDataPtr task_data) : Task(std::move(task_data)) {}
  bool PreProcessingImpl() override;
  bool ValidationImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

 private:
  std::vector<int> data;
  std::vector<int> labels;
  int m, n;
  boost::mpi::communicator world;
};
}  // namespace solovev_a_binary_image_marking
