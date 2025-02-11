#pragma once

#include <gtest/gtest.h>

#include <boost/mpi/collectives.hpp>
#include <boost/mpi/communicator.hpp>
#include <cmath>
#include <memory>
#include <numeric>
#include <vector>

#include "core/task/include/task.hpp"

namespace deryabin_m_cannons_algorithm_mpi {

class CannonsAlgorithmMPITaskSequential : public ppc::core::Task {
 public:
  explicit CannonsAlgorithmMPITaskSequential(ppc::core::TaskDataPtr task_data)
      : Task(std::move(task_data)) {}
  bool PreProcessingImpl() override;
  bool ValidationImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

 private:
  std::vector<double> input_matrix_A;
  std::vector<double> input_matrix_B;
  std::vector<double> output_matrix_C;
};
class CannonsAlgorithmMPITaskParallel : public ppc::core::Task {
 public:
  explicit CannonsAlgorithmMPITaskParallel(ppc::core::TaskDataPtr task_data)
      : Task(std::move(task_data)) {}
  bool PreProcessingImpl() override;
  bool ValidationImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

 private:
  std::vector<double> input_matrix_A, local_input_matrix_A;
  std::vector<double> input_matrix_B, local_input_matrix_B;
  std::vector<double> output_matrix_C, local_output_matrix_C;
  boost::mpi::communicator world;
};
}  // namespace deryabin_m_cannons_algorithm_mpi
