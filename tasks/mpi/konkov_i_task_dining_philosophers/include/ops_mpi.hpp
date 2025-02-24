#ifndef MODULES_TASK_2_KONKOV_I_DINING_PHILOSOPHERS_OPS_MPI_HPP
#define MODULES_TASK_2_KONKOV_I_DINING_PHILOSOPHERS_OPS_MPI_HPP

#include <vector>

namespace konkov_i_dining_philosophers {

class DiningPhilosophers {
 public:
  explicit DiningPhilosophers(int num_philosophers);
  [[nodiscard]] bool Validation() const;
  bool PreProcessing();
  bool Run();
  bool PostProcessing();
  int CheckDeadlock();

 private:
  int num_philosophers_;
  int rank_;
  int size_;
  std::vector<int> fork_states_;
  std::vector<int> philosopher_states_;

  void InitPhilosophers();
  void PhilosopherActions(int id);
  int IsDeadlock();
  void UpdateForkStates();
};

}  // namespace konkov_i_dining_philosophers

#endif  // MODULES_TASK_2_KONKOV_I_DINING_PHILOSOPHERS_OPS_MPI_HPP