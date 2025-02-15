#pragma once

#include <gtest/gtest.h>

#include <boost/mpi/collectives.hpp>
#include <boost/mpi/communicator.hpp>
#include <climits>
#include <memory>
#include <numeric>
#include <string>
#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

namespace kavtorev_d_most_different_neighbor_elements_mpi {

class most_different_neighbor_elements_seq : public ppc::core::Task {
 public:
  explicit most_different_neighbor_elements_seq(ppc::core::TaskDataPtr task_data) : Task(std::move(task_data)) {}
  bool PreProcessingImpl() override;
  bool ValidationImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

 private:
  std::vector<std::pair<int, int>> input_;
  std::pair<int, int> res{};
};

class most_different_neighbor_elements_mpi : public ppc::core::Task {
 public:
  explicit most_different_neighbor_elements_mpi(std::shared_ptr<ppc::core::TaskData> taskData_)
      : Task(std::move(taskData_)) {}
  bool PreProcessingImpl() override;
  bool ValidationImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

 private:
  std::vector<int> input_, local_input_;
  std::pair<int, int> res;
  size_t size;
  size_t st;
  boost::mpi::communicator world;
};

}  // namespace kavtorev_d_most_different_neighbor_elements_mpi