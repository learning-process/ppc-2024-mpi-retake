#pragma once

#include <boost/mpi/communicator.hpp>
#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

namespace veliev_e_sum_values_by_rows_matrix_mpi {
class SumValuesByRowsMatrixMpi : public ppc::core::Task {
 public:
  explicit SumValuesByRowsMatrixMpi(ppc::core::TaskDataPtr task_data) : Task(std::move(task_data)) {}
  bool PreProcessingImpl() override;
  bool ValidationImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

 private:
  std::vector<int> input_, output_;
  int elem_total_, cols_total_, rows_total_;
  boost::mpi::communicator world_;
};
void GetRndMatrix(std::vector<int>& vec);
void SeqProcForChecking(std::vector<int>& vec, int rows_size, std::vector<int>& output);
}  // namespace veliev_e_sum_values_by_rows_matrix_mpi