// Copyright 2023 Nesterov Alexander
#pragma once

#include <boost/mpi/collectives.hpp>
#include <boost/mpi/communicator.hpp>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

namespace shishkarev_a_sum_of_vector_elements_mpi {

class MPIVectorSumSequential : public ppc::core::Task {
 public:
  explicit MPIVectorSumSequential(std::shared_ptr<ppc::core::TaskData> task_data) : Task(std::move(task_data)) {}
  bool PreProcessingImpl() override;
  bool ValidationImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

 private:
  std::vector<int> input_vector_;
  int result_{};
  std::string operation_;
};

class MPIVectorSumParallel : public ppc::core::Task {
 public:
  explicit MPIVectorSumParallel(std::shared_ptr<ppc::core::TaskData> task_data) : Task(std::move(task_data)) {}
  bool PreProcessingImpl() override;
  bool ValidationImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

 private:
  std::vector<int> input_vector_, local_vector_;
  int result_{}, local_sum_;
  std::string operation_;
  boost::mpi::communicator world_;
};

}  // namespace shishkarev_a_sum_of_vector_elements_mpi
