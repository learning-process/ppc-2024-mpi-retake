// Copyright 2023 Nesterov Alexander
#pragma once

#include <boost/mpi/collectives.hpp>
#include <boost/mpi/communicator.hpp>
#include <cstdint>
#include <cstdlib>
#include <string>
#include <utility>

#include "core/task/include/task.hpp"

namespace makhov_m_monte_carlo_method_mpi {

class TestMPITaskParallel : public ppc::core::Task {
 public:
  explicit TestMPITaskParallel(ppc::core::TaskDataPtr task_data) : Task(std::move(task_data)) {}
  bool PreProcessingImpl() override;
  bool ValidationImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

 private:
  std::string funcStr_;
  double limits_[2]{};
  int numSamples_{};
  double globalSum_{};
  double answer_{};
  uint8_t* answerDataPtr_{};
  uint32_t dimension_{};
  boost::mpi::communicator world_;
};

}  // namespace makhov_m_monte_carlo_method_mpi