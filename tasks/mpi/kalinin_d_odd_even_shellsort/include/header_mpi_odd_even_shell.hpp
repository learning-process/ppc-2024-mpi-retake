#pragma once

#include <boost/mpi/communicator.hpp>
#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

namespace kalinin_d_odd_even_shell_mpi {
class OddEvenShellMpi : public ppc::core::Task {
 public:
  explicit OddEvenShellMpi(ppc::core::TaskDataPtr task_data) : Task(std::move(task_data)) {}
  bool PreProcessingImpl() override;
  bool ValidationImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;
  void PrepareInput(int &local_sz);
  void PerformOddEvenPhase(int phase, int id, int sz, bool is_even, std::vector<int> &local_vec, int local_sz);
  void ExchangeData(int id, int neighbour, std::vector<int> &local_vec, std::vector<int> &received_data, int send_tag,
                    int recv_tag);
  void GatherResults(int id, std::vector<int> &local_vec, int local_sz);

  static void ShellSort(std::vector<int> &vec);

 private:
  std::vector<int> input_;
  std::vector<int> output_;
  boost::mpi::communicator world_;
};
void GimmeRandVec(std::vector<int> &vec);

}  // namespace kalinin_d_odd_even_shell_mpi
