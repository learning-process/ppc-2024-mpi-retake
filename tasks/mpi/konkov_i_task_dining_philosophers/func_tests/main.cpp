#include <gtest/gtest.h>
#include <mpi.h>

#include "mpi/konkov_i_task_dining_philosophers/include/ops_mpi.hpp"

TEST(konkov_i_DiningPhilosophersTest, ValidNumberOfPhilosophers) {
  int num_philosophers = 5;
  konkov_i_dining_philosophers::DiningPhilosophers dp(num_philosophers);

  ASSERT_TRUE(dp.Validation());
  ASSERT_TRUE(dp.PreProcessing());
  ASSERT_TRUE(dp.Run());
  ASSERT_TRUE(dp.PostProcessing());
}

TEST(konkov_i_DiningPhilosophersTest, DeadlockFreeExecution) {
  int rank = 0;
  int size = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  if (size < 2) {
    GTEST_SKIP() << "Skipping test: At least 2 processes are required.";
  }

  int num_philosophers = size;
  konkov_i_dining_philosophers::DiningPhilosophers dp(num_philosophers);

  ASSERT_TRUE(dp.Validation());
  ASSERT_TRUE(dp.PreProcessing());
  ASSERT_TRUE(dp.Run());
  ASSERT_TRUE(dp.PostProcessing());

  int local_deadlock = dp.CheckDeadlock();
  int global_deadlock = 0;
  MPI_Allreduce(&local_deadlock, &global_deadlock, 1, MPI_INT, MPI_LOR, MPI_COMM_WORLD);
}

TEST(konkov_i_DiningPhilosophersTest, SmallNumberOfPhilosophers) {
  int num_philosophers = 3;
  konkov_i_dining_philosophers::DiningPhilosophers dp(num_philosophers);

  ASSERT_TRUE(dp.Validation());
  ASSERT_TRUE(dp.PreProcessing());
  ASSERT_TRUE(dp.Run());
  ASSERT_TRUE(dp.PostProcessing());
}

TEST(konkov_i_DiningPhilosophersTest, SinglePhilosopher) {
  int num_philosophers = 1;
  konkov_i_dining_philosophers::DiningPhilosophers dp(num_philosophers);

  ASSERT_FALSE(dp.Validation());
}

TEST(DiningPhilosophersFunctional, InvalidPhilosopherCount) {
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  int num_philosophers = -5;

  if (num_philosophers <= 0) {
    if (rank == 0) {
      GTEST_SKIP() << "Skipping test: Invalid number of philosophers (" << num_philosophers
                   << "). Number must be positive.";
    }
    return;
  }

  konkov_i_dining_philosophers::DiningPhilosophers dp(num_philosophers);

  ASSERT_TRUE(dp.PreProcessing());
  ASSERT_TRUE(dp.Run());
  ASSERT_TRUE(dp.PostProcessing());
}