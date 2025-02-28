#include <gtest/gtest.h>

#include <boost/mpi.hpp>
#include <chrono>

#include "mpi/konkov_i_task_dining_philosophers_mpi/include/ops_mpi.hpp"

TEST(DiningPhilosophersPerfTest, PerformanceWith10Philosophers) {
  boost::mpi::environment env;
  boost::mpi::communicator world;

  const int NUM_PHILOSOPHERS = 10;
  konkov_i_task_dp::DiningPhilosophersMPI philosophers(NUM_PHILOSOPHERS);

  auto start = std::chrono::high_resolution_clock::now();
  philosophers.PreProcessing();
  philosophers.Run();
  philosophers.PostProcessing();
  auto end = std::chrono::high_resolution_clock::now();

  EXPECT_LT(std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count(), 5000);
}

TEST(DiningPhilosophersPerfTest, ScalabilityWith50Philosophers) {
  boost::mpi::environment env;
  boost::mpi::communicator world;

  const int NUM_PHILOSOPHERS = 50;
  konkov_i_task_dp::DiningPhilosophersMPI philosophers(NUM_PHILOSOPHERS);

  auto start = std::chrono::high_resolution_clock::now();
  philosophers.PreProcessing();
  philosophers.Run();
  philosophers.PostProcessing();
  auto end = std::chrono::high_resolution_clock::now();

  EXPECT_LT(std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count(), 10000);
}

TEST(DiningPhilosophersPerfTest, MinimalValidCase) {
  boost::mpi::environment env;
  boost::mpi::communicator world;

  konkov_i_task_dp::DiningPhilosophersMPI philosophers(2);

  auto start = std::chrono::high_resolution_clock::now();
  philosophers.PreProcessing();
  philosophers.Run();
  philosophers.PostProcessing();
  auto end = std::chrono::high_resolution_clock::now();

  EXPECT_TRUE(philosophers.Validation());
  EXPECT_LT(std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count(), 2000);
}