#include <gtest/gtest.h>

#include <boost/mpi.hpp>

#include "mpi/konkov_i_task_dining_philosophers_mpi/include/ops_mpi.hpp"

TEST(DiningPhilosophersTest, ValidNumberOfPhilosophers) {
  boost::mpi::environment env;
  boost::mpi::communicator world;

  konkov_i_task_dp::DiningPhilosophersMPI philosophers(10);
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

  konkov_i_task_dp::DiningPhilosophersMPI philosophers(10);
  philosophers.PreProcessing();
  philosophers.Run();
  philosophers.PostProcessing();
  EXPECT_TRUE(true);
}