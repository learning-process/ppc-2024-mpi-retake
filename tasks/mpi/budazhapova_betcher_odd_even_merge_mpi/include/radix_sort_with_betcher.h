#pragma once

#include <boost/mpi/collectives.hpp>
#include <boost/mpi/communicator.hpp>
#include <memory>
#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

namespace budazhapova_betcher_odd_even_merge_mpi {

class MergeSequential : public ppc::core::Task {
 public:
  explicit MergeSequential(std::shared_ptr<ppc::core::TaskData> task_data) : Task(std::move(task_data)) {}
  bool PreProcessingImpl() override;
  bool ValidationImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

 private:
  std::vector<int> res_;
  std::vector<int> local_res_;
  int n_el_ = 0;
};
class MergeParallel : public ppc::core::Task {
 public:
  explicit MergeParallel(std::shared_ptr<ppc::core::TaskData> task_data) : Task(std::move(task_data)) {}
  bool PreProcessingImpl() override;
  bool ValidationImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

 private:
  std::vector<int> res_;
  std::vector<int> local_res_;
  std::vector<int> fin_res_;
  int n_el_ = 0;

  boost::mpi::communicator world_;
};
}  // namespace budazhapova_betcher_odd_even_merge_mpi
