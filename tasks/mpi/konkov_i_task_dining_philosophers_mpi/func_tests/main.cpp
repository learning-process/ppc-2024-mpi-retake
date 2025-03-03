#include <gtest/gtest.h>

#include <boost/mpi/collectives.hpp>
#include <boost/mpi/communicator.hpp>
#include <memory>
#include <vector>

#include "core/task/include/task.hpp"
#include "mpi/konkov_i_task_dining_philosophers_mpi/include/ops_mpi.hpp"

TEST(konkov_i_task_dining_philosophers_mpi, test_valid_number_of_philosophers) {
  boost::mpi::communicator world;
  auto task_data = std::make_shared<ppc::core::TaskData>();
  int count_philosophers = 4;

  if (world.rank() == 0) {
    task_data->inputs_count.push_back(count_philosophers);
  }

  konkov_i_task_dining_philosophers_mpi::DiningPhilosophersMPI dining_philosophers_mpi(task_data);

  if (world.size() < 2) {
    GTEST_SKIP();
  }

  ASSERT_TRUE(dining_philosophers_mpi.ValidationImpl());
}

TEST(konkov_i_task_dining_philosophers_mpi, test_invalid_philosopher_count) {
  boost::mpi::communicator world;

  if (world.rank() == 0) {
    int count_philosophers = -5;
    auto task_data = std::make_shared<ppc::core::TaskData>();
    task_data->inputs_count.push_back(count_philosophers);

    konkov_i_task_dining_philosophers_mpi::DiningPhilosophersMPI dining_philosophers_mpi(task_data);

    ASSERT_FALSE(dining_philosophers_mpi.ValidationImpl());
  }
}

TEST(konkov_i_task_dining_philosophers_mpi, test_deadlock_free_execution) {
  boost::mpi::communicator world;
  auto task_data = std::make_shared<ppc::core::TaskData>();
  int count_philosophers = world.size();

  if (world.rank() == 0) {
    task_data->inputs_count.push_back(count_philosophers);
  }

  konkov_i_task_dining_philosophers_mpi::DiningPhilosophersMPI dining_philosophers_mpi(task_data);

  if (world.size() < 2) {
    GTEST_SKIP();
  }

  ASSERT_TRUE(dining_philosophers_mpi.ValidationImpl());
  dining_philosophers_mpi.PreProcessingImpl();
  dining_philosophers_mpi.RunImpl();
  dining_philosophers_mpi.PostProcessingImpl();
}