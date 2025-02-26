#ifndef KONKOV_I_TASK_DINING_PHILOSOPHERS_OPS_MPI_HPP
#define KONKOV_I_TASK_DINING_PHILOSOPHERS_OPS_MPI_HPP

#include <mpi.h>

#include <vector>

namespace dining_philosophers {

class DiningPhilosophersMPI {
 public:
  DiningPhilosophersMPI(int num_philosophers);
  ~DiningPhilosophersMPI();

  void Validation();
  void PreProcessing();
  void Run();
  void PostProcessing();

 private:
  int num_philosophers;
  int rank, size;
  MPI_Comm comm;
  std::vector<int> states;

  void Think(int philosopher_id);
  void Eat(int philosopher_id);
  void TakeForks(int philosopher_id);
  void PutForks(int philosopher_id);
};

}  // namespace dining_philosophers

#endif  // KONKOV_I_TASK_DINING_PHILOSOPHERS_OPS_MPI_HPP
