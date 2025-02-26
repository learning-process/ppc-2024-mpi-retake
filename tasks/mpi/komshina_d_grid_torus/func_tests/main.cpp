#include <gtest/gtest.h>

#include <boost/mpi/communicator.hpp>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "core/task/include/task.hpp"
#include "mpi/komshina_d_grid_torus/include/ops_mpi.hpp"

namespace komshina_d_grid_torus_mpi {

TEST(komshina_d_grid_torus_mpi, validation_failed_wrong_task_data) {
  boost::mpi::communicator world;
  std::shared_ptr<ppc::core::TaskData> task_data_mpi = std::make_shared<ppc::core::TaskData>();
  TestTaskMPI torus(task_data_mpi);
  if (world.rank() == 0) {
    ASSERT_FALSE(torus.ValidationImpl());
  }
}

TEST(komshina_d_grid_torus_mpi, check_out_of_bounds_destination) {
  boost::mpi::communicator world;
  if (world.size() < 4) {
    GTEST_SKIP();
    return;
  }

  komshina_d_grid_torus_mpi::TestTaskMPI::TaskData in("out of bounds", -1);
  komshina_d_grid_torus_mpi::TestTaskMPI::TaskData out;

  std::shared_ptr<ppc::core::TaskData> task_data_mpi = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    task_data_mpi->inputs.emplace_back(reinterpret_cast<uint8_t*>(&in));
    task_data_mpi->inputs_count.emplace_back(1);
    task_data_mpi->outputs.emplace_back(reinterpret_cast<uint8_t*>(&out));
    task_data_mpi->outputs_count.emplace_back(1);
  }

  TestTaskMPI torus(task_data_mpi);
  ASSERT_FALSE(torus.PreProcessingImpl());
}

TEST(komshina_d_grid_torus_mpi, large_grid_data_send_receive) {
  boost::mpi::communicator world;
  if (world.size() < 100) {
    GTEST_SKIP();
    return;
  }

  komshina_d_grid_torus_mpi::TestTaskMPI::TaskData in("large grid test", 80);
  komshina_d_grid_torus_mpi::TestTaskMPI::TaskData out;

  std::shared_ptr<ppc::core::TaskData> task_data_mpi = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    task_data_mpi->inputs.emplace_back(reinterpret_cast<uint8_t*>(&in));
    task_data_mpi->inputs_count.emplace_back(1);
    task_data_mpi->outputs.emplace_back(reinterpret_cast<uint8_t*>(&out));
    task_data_mpi->outputs_count.emplace_back(1);
  }

  TestTaskMPI torus(task_data_mpi);
  ASSERT_TRUE(torus.ValidationImpl());
  ASSERT_TRUE(torus.PreProcessingImpl());

  torus.RunImpl();
  torus.PostProcessingImpl();

  if (world.rank() == 0) {
    ASSERT_EQ(out.payload, in.payload);
    ASSERT_EQ(out.target, in.target);
  }
}

TEST(komshina_d_grid_torus_mpi, route_calculation_same_source_and_destination) {
  boost::mpi::communicator world;
  if (world.size() < 4) {
    GTEST_SKIP();
    return;
  }

  int source_rank = 0;
  int dest_rank = 0;

  std::vector<int> expected_route = {source_rank};

  TestTaskMPI torus(std::make_shared<ppc::core::TaskData>());
  std::vector<int> route = komshina_d_grid_torus_mpi::TestTaskMPI::CalculateRoute(dest_rank, 3, 3);
  ASSERT_EQ(route, expected_route);
}

}  // namespace komshina_d_grid_torus_mpi