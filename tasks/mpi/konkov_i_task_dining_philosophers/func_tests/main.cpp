#include <gtest/gtest.h>
#include <mpi.h>

#include "mpi/konkov_i_task_dining_philosophers/include/ops_mpi.hpp"

TEST(DiningPhilosophersTest, ValidNumberOfPhilosophers) { EXPECT_NO_THROW(DiningPhilosophersMPI(5).Validation()); }

TEST(DiningPhilosophersTest, InvalidPhilosopherCount) {
  EXPECT_THROW(DiningPhilosophersMPI(0).Validation(), std::invalid_argument);
}

TEST(DiningPhilosophersTest, DeadlockFreeExecution) {
  DiningPhilosophersMPI philosophers(5);
  philosophers.Validation();
  philosophers.PreProcessing();
  EXPECT_NO_THROW(philosophers.Run());
  philosophers.PostProcessing();
}

TEST(DiningPhilosophersTest, SmallNumberOfPhilosophers) {
  DiningPhilosophersMPI philosophers(2);
  philosophers.Validation();
  philosophers.PreProcessing();
  EXPECT_NO_THROW(philosophers.Run());
  philosophers.PostProcessing();
}

TEST(DiningPhilosophersTest, SinglePhilosopher) {
  DiningPhilosophersMPI philosophers(1);
  philosophers.Validation();
  philosophers.PreProcessing();
  EXPECT_NO_THROW(philosophers.Run());
  philosophers.PostProcessing();
}
