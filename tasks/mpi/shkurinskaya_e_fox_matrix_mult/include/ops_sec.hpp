#pragma once

#include <boost/mpi/collectives.hpp>
#include <boost/mpi/communicator.hpp>
#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

namespace shkurinskaya_e_fox_mat_mul_mpi {
std::vector<double> getRandomMatrix(int rows, int cols);
class FoxMatMulMPI : public ppc::core::Task {
 public:
  explicit FoxMatMulMPI(ppc::core::TaskDataPtr task_data) : Task(std::move(task_data)) {}
  bool PreProcessingImpl() override;
  bool ValidationImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

 private:
  std::vector<double> inputA, inputB, output, localA, localB;
  int matrix_size, root, sz, block_sz;
  boost::mpi::communicator world;
};

}  // namespace shkurinskaya_e_fox_mat_mul_mpi