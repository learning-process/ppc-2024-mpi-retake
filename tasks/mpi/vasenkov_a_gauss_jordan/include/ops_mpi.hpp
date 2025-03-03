#include <gtest/gtest.h>

#include <boost/mpi/collectives.hpp>
#include <boost/mpi/communicator.hpp>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

namespace vasenkov_a_gauss_jordan_mpi {

class GaussJordanMethodParallelMPI : public ppc::core::Task {
  private:
  std::vector<double> matrix;
  bool solve = true;
  int n;
  boost::mpi::communicator world;
public:
  explicit GaussJordanMethodParallelMPI(
      std::shared_ptr<ppc::core::TaskData> taskData_)
      : Task(std::move(taskData_)) {}
  bool PreProcessingImpl() override;
  bool ValidationImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;


};

class GaussJordanMethodSequentialMPI : public ppc::core::Task {
private:
  int n_size;
  std::vector<double> sys_matrix;

public:
  explicit GaussJordanMethodSequentialMPI(ppc::core::TaskDataPtr task_data)
      : Task(std::move(task_data)) {}
  bool PreProcessingImpl() override;
  bool ValidationImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;
};
} // namespace vasenkov_a_gauss_jordan_mpi