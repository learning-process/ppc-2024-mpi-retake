#include <gtest/gtest.h>

#include "mpi/konkov_i_task_dining_philosophers/include/ops_mpi.hpp"

TEST(DiningPhilosophersTest, ValidNumberOfPhilosophers) {
  int argc = 0;
  char** argv = nullptr;
  MPI_Init(&argc, &argv);

  dining_philosophers::DiningPhilosophersMPI philosophers(5);
  EXPECT_NO_THROW(philosophers.Validation());

  MPI_Finalize();
}

TEST(DiningPhilosophersTest, DeadlockFreeExecution) {
  int argc = 0;
  char** argv = nullptr;
  MPI_Init(&argc, &argv);

  dining_philosophers::DiningPhilosophersMPI philosophers(5);
  philosophers.PreProcessing();
  philosophers.Run();
  philosophers.PostProcessing();

  EXPECT_TRUE(true);

  MPI_Finalize();
}

TEST(DiningPhilosophersTest, SmallNumberOfPhilosophers) {
  int argc = 0;
  char** argv = nullptr;
  MPI_Init(&argc, &argv);

  dining_philosophers::DiningPhilosophersMPI philosophers(2);
  philosophers.PreProcessing();
  philosophers.Run();
  philosophers.PostProcessing();

  EXPECT_TRUE(true);

  MPI_Finalize();
}

TEST(DiningPhilosophersTest, SinglePhilosopher) {
  int argc = 0;
  char** argv = nullptr;
  MPI_Init(&argc, &argv);

  dining_philosophers::DiningPhilosophersMPI philosophers(1);
  philosophers.PreProcessing();
  philosophers.Run();
  philosophers.PostProcessing();

  EXPECT_TRUE(true);

  MPI_Finalize();
}

TEST(DiningPhilosophersTest, InvalidPhilosopherCount) {
  int argc = 0;
  char** argv = nullptr;
  MPI_Init(&argc, &argv);

  dining_philosophers::DiningPhilosophersMPI philosophers(0);
  EXPECT_EXIT(philosophers.Validation(), ::testing::ExitedWithCode(1), "Error: Not enough processes for philosophers.");

  MPI_Finalize();
}
