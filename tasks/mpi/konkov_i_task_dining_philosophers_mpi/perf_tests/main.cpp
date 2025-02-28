#include <gtest/gtest.h>

#include <boost/mpi.hpp>
#include <chrono>

#include "mpi/konkov_i_task_dining_philosophers_mpi/include/ops_mpi.hpp"

TEST(DiningPhilosophersPerfTest, test_pipeline_run_mpi) {
  boost::mpi::environment env;
  boost::mpi::communicator world;

  konkov_i_task_dp::DiningPhilosophersMPI philosophers(world.size() - 1);

  auto start = std::chrono::high_resolution_clock::now();
  philosophers.PreProcessing();
  philosophers.Run();
  philosophers.PostProcessing();
  auto end = std::chrono::high_resolution_clock::now();

  EXPECT_LT(std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count(), 5000);
}

TEST(DiningPhilosophersPerfTest, test_task_run_mpi) {
  boost::mpi::environment env;
  boost::mpi::communicator world;

  konkov_i_task_dp::DiningPhilosophersMPI philosophers(world.size() - 1);

  auto start = std::chrono::high_resolution_clock::now();
  philosophers.PreProcessing();
  philosophers.Run();
  philosophers.PostProcessing();
  auto end = std::chrono::high_resolution_clock::now();

  EXPECT_LT(std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count(), 5000);
}