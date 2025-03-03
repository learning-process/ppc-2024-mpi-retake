#pragma once

#include <boost/mpi/collectives.hpp>
#include <boost/mpi/communicator.hpp>
#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

namespace shkurinskaya_e_fox_mat_mul_mpi {
std::vector<double> GetRandomMatrix(int rows, int cols);
class FoxMatMulMPI : public ppc::core::Task {
 public:
  explicit FoxMatMulMPI(ppc::core::TaskDataPtr task_data) : Task(std::move(task_data)) {}
  bool PreProcessingImpl() override;
  bool ValidationImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

 private:
  std::vector<double> inputA_, inputB_, output_;
  int matrix_size_, root_, sz_, block_sz_;
  boost::mpi::communicator world_;
};

}  // namespace shkurinskaya_e_fox_mat_mul_mpi