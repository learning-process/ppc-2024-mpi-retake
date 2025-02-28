#ifndef MPI_KONKOV_I_TASK_DINING_PHILOSOPHERS_INCLUDE_OPS_MPI_HPP_
#define MPI_KONKOV_I_TASK_DINING_PHILOSOPHERS_INCLUDE_OPS_MPI_HPP_

#include <boost/mpi.hpp>
#include <queue>
#include <vector>

namespace konkov_i_task_dp {

class DiningPhilosophersMPI {
 public:
  explicit DiningPhilosophersMPI(int num_philosophers);
  void PreProcessing();
  bool Validation();
  void Run();
  void PostProcessing();

 private:
  void PhilosopherProcess(int worker_id);
  void ForkManagementProcess();
  std::pair<int, int> getAssignedPhilosophers(int worker_id) const;

  int num_philosophers_;
  boost::mpi::communicator world_;
  std::vector<int> fork_status_;
};

}  // namespace konkov_i_task_dp

#endif  // MPI_KONKOV_I_TASK_DINING_PHILOSOPHERS_INCLUDE_OPS_MPI_HPP_