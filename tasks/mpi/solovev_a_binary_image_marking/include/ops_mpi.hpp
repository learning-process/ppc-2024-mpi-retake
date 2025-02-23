#pragma once

#include <boost/mpi/collectives.hpp>
#include <boost/mpi/communicator.hpp>
#include <cstring>
#include <memory>
#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

namespace solovev_a_binary_image_marking {

struct Point {
  int x, y;
};

class TestMPITaskSequential : public ppc::core::Task {
 public:
  explicit TestMPITaskSequential(std::shared_ptr<ppc::core::TaskData> task_data) : Task(std::move(task_data)) {}
  bool PreProcessingImpl() override;
  bool ValidationImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

 private:
  std::vector<int> data_seq_;
  std::vector<int> labels_seq_;
  int m_seq_, n_seq_;
};

class TestMPITaskParallel : public ppc::core::Task {
 public:
  explicit TestMPITaskParallel(ppc::core::TaskDataPtr task_data) : Task(std::move(task_data)) {}
  bool PreProcessingImpl() override;
  bool ValidationImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

 private:
  std::vector<int> data_;
  std::vector<int> labels_;
  int m_, n_;
  boost::mpi::communicator world_;
};
}  // namespace solovev_a_binary_image_marking