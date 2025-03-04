#include <gtest/gtest.h>

#include <boost/mpi/collectives.hpp>
#include <boost/mpi/communicator.hpp>
#include <climits>
#include <cstdint>
#include <memory>
#include <vector>

#include "core/task/include/task.hpp"
#include "mpi/sedova_o_linear_topology/include/ops_mpi.hpp"

TEST(sedova_o_linear_topology_mpi, test_1000) {
  boost::mpi::communicator world;
  std::vector<int> input;
  bool result = false;
  int count = 1000;

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> task_data_par = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    input = sedova_o_linear_topology_mpi::GetRandomVector(count);
    task_data_par->inputs.emplace_back(reinterpret_cast<uint8_t*>(input.data()));
    task_data_par->inputs_count.emplace_back(input.size());

    task_data_par->outputs.emplace_back(reinterpret_cast<uint8_t*>(&result));
    task_data_par->outputs_count.emplace_back(1);
  }

  sedova_o_linear_topology_mpi::TestTaskMPI test_task_parallel(task_data_par);
  ASSERT_EQ(test_task_parallel.ValidationImpl(), true);
  test_task_parallel.PreProcessingImpl();
  test_task_parallel.RunImpl();
  test_task_parallel.PostProcessingImpl();

  if (world.rank() == 0) {
    ASSERT_TRUE(result);
  }
}

TEST(sedova_o_linear_topology_mpi, test_10000) {
  boost::mpi::communicator world;
  std::vector<int> input;
  bool result = false;
  int count = 10000;

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> task_data_par = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    input = sedova_o_linear_topology_mpi::GetRandomVector(count);
    task_data_par->inputs.emplace_back(reinterpret_cast<uint8_t*>(input.data()));
    task_data_par->inputs_count.emplace_back(input.size());

    task_data_par->outputs.emplace_back(reinterpret_cast<uint8_t*>(&result));
    task_data_par->outputs_count.emplace_back(1);
  }

  sedova_o_linear_topology_mpi::TestTaskMPI test_task_parallel(task_data_par);
  ASSERT_EQ(test_task_parallel.ValidationImpl(), true);
  test_task_parallel.PreProcessingImpl();
  test_task_parallel.RunImpl();
  test_task_parallel.PostProcessingImpl();

  if (world.rank() == 0) {
    ASSERT_TRUE(result);
  }
}
