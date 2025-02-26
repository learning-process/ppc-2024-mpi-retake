#include <gtest/gtest.h>
#include <mpi.h>

#include "mpi/konkov_i_task_dining_philosophers/include/ops_mpi.hpp"

TEST(DiningPhilosophersPerf, test_pipeline_run_mpi) {
  DiningPhilosophersMPI philosophers(1000);
  philosophers.Validation();
  philosophers.PreProcessing();
  philosophers.Run();
  philosophers.PostProcessing();
}

TEST(DiningPhilosophersPerf, test_task_run_mpi) {
  DiningPhilosophersMPI philosophers(2000);
  philosophers.Validation();
  philosophers.PreProcessing();
  philosophers.Run();
  philosophers.PostProcessing();
}
