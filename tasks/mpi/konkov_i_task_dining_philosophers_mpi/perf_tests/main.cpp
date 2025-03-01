#include <memory>

#include "boost/mpi/communicator.hpp"
#include "core/task/include/task.hpp"
#include "mpi/konkov_i_task_dining_philosophers_mpi/include/ops_mpi.hpp"
#include "gtest/gtest.h"

TEST(DiningPhilosophersMPIPerfTest, test_pipeline_run_mpi) {
  boost::mpi::communicator world;
  if (world.rank() == 0) {
    ppc::core::TaskDataPtr task_data = std::make_shared<ppc::core::TaskData>();
    task_data->inputs_count = {100};

    konkov_i_task_dining_philosophers_mpi::DiningPhilosophersMPI task(task_data);
    task.PreProcessingImpl();
    task.ValidationImpl();
    task.RunImpl();
    task.PostProcessingImpl();
  }
}

TEST(DiningPhilosophersMPIPerfTest, test_task_run_mpi) {
  boost::mpi::communicator world;
  if (world.rank() == 0) {
    ppc::core::TaskDataPtr task_data = std::make_shared<ppc::core::TaskData>();
    task_data->inputs_count = {200};

    konkov_i_task_dining_philosophers_mpi::DiningPhilosophersMPI task(task_data);
    task.PreProcessingImpl();
    task.ValidationImpl();
    task.RunImpl();
    task.PostProcessingImpl();
  }
}
