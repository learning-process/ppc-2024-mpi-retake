#include <gtest/gtest.h>

#include <memory>

#include "core/task/include/task.hpp"
#include "mpi/konkov_i_task_dining_philosophers_mpi/include/ops_mpi.hpp"

TEST(konkov_i_task_dining_philosophers_mpi, ValidNumberOfPhilosophers) {
  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs_count.push_back(5);

  konkov_i_task_dining_philosophers_mpi::DiningPhilosophersMPI task(task_data);
  ASSERT_TRUE(task.Validation());
}

TEST(konkov_i_task_dining_philosophers_mpi, InvalidPhilosopherCount) {
  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs_count.push_back(1);

  konkov_i_task_dining_philosophers_mpi::DiningPhilosophersMPI task(task_data);
  ASSERT_FALSE(task.Validation());
}

TEST(konkov_i_task_dining_philosophers_mpi, DeadlockFreeExecution) {
  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs_count.push_back(4);

  konkov_i_task_dining_philosophers_mpi::DiningPhilosophersMPI task(task_data);
  ASSERT_TRUE(task.PreProcessing());
  ASSERT_TRUE(task.Validation());
  ASSERT_TRUE(task.Run());
  ASSERT_TRUE(task.PostProcessing());
}
