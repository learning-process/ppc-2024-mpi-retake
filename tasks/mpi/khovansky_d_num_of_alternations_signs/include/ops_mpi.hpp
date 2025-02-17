#pragma once

#include <boost/mpi/communicator.hpp>
#include <vector>

#include "core/task/include/task.hpp"

namespace khovansky_d_num_of_alternations_signs_mpi {

class NumOfAlternationsSignsSeq : public ppc::core::Task {
 public:
  explicit NumOfAlternationsSignsSeq(ppc::core::TaskDataPtr task_data) : Task(std::move(task_data)) {}
  bool PreProcessingImpl() override;
  bool ValidationImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

 private:
  std::vector<int> input;
  int res{};
};

class NumOfAlternationsSignsMpi : public ppc::core::Task {
 public:
  explicit NumOfAlternationsSignsMpi(ppc::core::TaskDataPtr task_data) : Task(std::move(task_data)) {}
  bool PreProcessingImpl() override;
  bool ValidationImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

 private:
  std::vector<int> input;
  std::vector<int> start;
  int res{};
  boost::mpi::communicator world;
};
}  // namespace khovansky_d_num_of_alternations_signs_mpi