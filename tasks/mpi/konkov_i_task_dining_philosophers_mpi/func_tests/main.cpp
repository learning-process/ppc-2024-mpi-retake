#include <gtest/gtest.h>

#include <boost/mpi.hpp>

#include "mpi/konkov_i_task_dining_philosophers_mpi/include/ops_mpi.hpp"

TEST(DiningPhilosophersTest, ValidNumberOfPhilosophers) {
  boost::mpi::environment env;
  boost::mpi::communicator world;

  konkov_i_task_dp::DiningPhilosophersMPI philosophers(world.size() - 1);
  EXPECT_TRUE(philosophers.Validation());
}

TEST(DiningPhilosophersTest, InvalidPhilosopherCount) {
  boost::mpi::environment env;
  boost::mpi::communicator world;

  konkov_i_task_dp::DiningPhilosophersMPI philosophers(1);
  EXPECT_FALSE(philosophers.Validation());
}

TEST(DiningPhilosophersTest, DeadlockFreeExecution) {
  boost::mpi::environment env;
  boost::mpi::communicator world;

  if (world.size() < 2) {
    GTEST_SKIP() << "This test requires at least 2 processes";
  }

  konkov_i_task_dp::DiningPhilosophersMPI philosophers(world.size() - 1);
  philosophers.PreProcessing();
  philosophers.Run();
  philosophers.PostProcessing();

  world.barrier();
}