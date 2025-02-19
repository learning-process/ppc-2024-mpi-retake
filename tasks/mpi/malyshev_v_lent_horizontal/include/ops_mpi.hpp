#pragma once

#include <boost/mpi/collectives.hpp>
#include <boost/mpi/communicator.hpp>
#include <random>
#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

namespace malyshev_v_lent_horizontal {

std::vector<int> GetRandomMatrix(int rows, int cols) {
  std::random_device dev;
  std::mt19937 gen(dev());
  std::vector<int> matrix(rows * cols);
  for (int i = 0; i < rows * cols; i++) {
    matrix[i] = gen() % 100;
  }
  return matrix;
}

std::vector<int> GetRandomVector(int size) {
  std::random_device dev;
  std::mt19937 gen(dev());
  std::vector<int> vector(size);
  for (int i = 0; i < size; i++) {
    vector[i] = gen() % 100;
  }
  return vector;
}

class MatVecMultMpi : public ppc::core::Task {
 public:
  explicit MatVecMultMpi(ppc::core::TaskDataPtr task_data) : Task(std::move(task_data)) {}
  bool PreProcessingImpl() override;
  bool ValidationImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

 private:
  std::vector<int> matrix_, vector_, local_matrix_, local_result_;
  unsigned int rows_, cols_;
  boost::mpi::communicator world_;
};

}  // namespace malyshev_v_lent_horizontal