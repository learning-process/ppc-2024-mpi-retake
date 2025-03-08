#include <gtest/gtest.h>

#include <boost/mpi/communicator.hpp>
#include <cstdint>
#include <memory>
#include <vector>

#include "core/task/include/task.hpp"
#include "mpi/dudchenko_o_sleeping_barber/include/ops_mpi.hpp"

TEST(dudchenko_o_sleeping_barber_mpi, validation_test_1) {
  boost::mpi::communicator world;

  std::shared_ptr<ppc::core::TaskData> task_data_par = std::make_shared<ppc::core::TaskData>();
  dudchenko_o_sleeping_barber_mpi::TestMPISleepingBarber test_mpi_task_parallel(task_data_par);

  if (world.rank() == 0) {
    task_data_par->inputs_count = {0};
    EXPECT_FALSE(test_mpi_task_parallel.ValidationImpl());
  }
}

TEST(dudchenko_o_sleeping_barber_mpi, validation_test_2) {
  boost::mpi::communicator world;

  std::shared_ptr<ppc::core::TaskData> task_data_par = std::make_shared<ppc::core::TaskData>();
  dudchenko_o_sleeping_barber_mpi::TestMPISleepingBarber test_mpi_task_parallel(task_data_par);

  if (world.rank() == 0) {
    task_data_par->inputs_count = {1};
    EXPECT_TRUE(test_mpi_task_parallel.ValidationImpl());
  }
}

TEST(dudchenko_o_sleeping_barber_mpi, validation_test_3) {
  boost::mpi::communicator world;

  std::shared_ptr<ppc::core::TaskData> task_data_par = std::make_shared<ppc::core::TaskData>();
  dudchenko_o_sleeping_barber_mpi::TestMPISleepingBarber test_mpi_task_parallel(task_data_par);

  if (world.rank() == 0) {
    task_data_par->inputs_count = {1};
    EXPECT_TRUE(test_mpi_task_parallel.ValidationImpl());
  }
}

namespace {
void RunSleepingBarberTest(int max_waiting_chairs) {
  boost::mpi::communicator world;

  if (world.size() <= 1) {
    return;
  }

  auto task_data_par = std::make_shared<ppc::core::TaskData>();
  int global_res = -1;

  task_data_par->inputs_count.emplace_back(max_waiting_chairs);
  task_data_par->outputs.emplace_back(reinterpret_cast<uint8_t*>(&global_res));
  task_data_par->outputs_count.emplace_back(sizeof(global_res));

  dudchenko_o_sleeping_barber_mpi::TestMPISleepingBarber test_mpi_task_parallel(task_data_par);

  ASSERT_TRUE(test_mpi_task_parallel.ValidationImpl());
  ASSERT_TRUE(test_mpi_task_parallel.PreProcessingImpl());
  ASSERT_TRUE(test_mpi_task_parallel.RunImpl());
  ASSERT_TRUE(test_mpi_task_parallel.PostProcessingImpl());

  world.barrier();

  if (world.rank() == 0) {
    EXPECT_EQ(global_res, 0);
  }
}
}  // namespace

TEST(dudchenko_o_sleeping_barber_mpi, functional_test_small) { RunSleepingBarberTest(1); }

TEST(dudchenko_o_sleeping_barber_mpi, functional_test_medium) { RunSleepingBarberTest(3); }

TEST(dudchenko_o_sleeping_barber_mpi, functional_test_large) { RunSleepingBarberTest(999); }
