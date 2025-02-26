#ifndef KONKOV_I_TASK_DINING_PHILOSOPHERS_OPS_MPI_HPP
#define KONKOV_I_TASK_DINING_PHILOSOPHERS_OPS_MPI_HPP

#include <mpi.h>

#include <vector>

class DiningPhilosophersMPI {
 public:
  explicit DiningPhilosophersMPI(int numPhilosophers);

  void Validation();
  void PreProcessing();
  void Run();
  void PostProcessing();

 private:
  int numPhilosophers;
  int rank;
  int size;
  int localStart, localEnd;
  MPI_Comm comm;

  void PickUpForks(int id);
  void PutDownForks(int id);
  void Think(int id);
  void Eat(int id);
};

#endif  // KONKOV_I_TASK_DINING_PHILOSOPHERS_OPS_MPI_HPP
