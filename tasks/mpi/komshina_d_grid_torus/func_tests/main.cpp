#include <gtest/gtest.h>

#include <cstdint>
#include <random>
#include <vector>
#include <string>
#include <memory>
#include <boost/mpi/communicator.hpp>

#include "core/task/include/task.hpp"
#include "core/util/include/util.hpp"
#include "mpi/komshina_d_grid_torus/include/ops_mpi.hpp"

namespace {
static std::vector<int> getRandomVector(int sz, int a, int b) {
  std::random_device dev;
  std::mt19937 gen(dev());
  std::vector<int> vec(sz);
  for (int i = 0; i < sz; i++) {
    vec[i] = static_cast<int>(gen() % (b - a + 1) + a);
  }
  return vec;
}
}  // namespace

TEST(komshina_d_grid_torus_mpi, Test_Negative_Validation) {
  boost::mpi::communicator world;
  std::vector<int> global_vec;
  std::vector<int> global_path;

  std::shared_ptr<ppc::core::TaskData> task_data_mpi = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    global_vec = getRandomVector(1, 0, 100);
    task_data_mpi->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vec.data()));
    task_data_mpi->inputs_count.emplace_back(global_vec.size());

    global_path.push_back(0);
    task_data_mpi->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_path.data()));
    task_data_mpi->outputs_count.emplace_back(global_path.size());
  }

  komshina_d_grid_torus_mpi::TestTaskMPI test_task(task_data_mpi);
  ASSERT_EQ(test_task.ValidationImpl(), false);
}

TEST(komshina_d_grid_torus_mpi, check_world_size_is_even) {
  boost::mpi::communicator world;
  if (world.size() < 4) {
    GTEST_SKIP();
  }

  ASSERT_TRUE(world.size() % 2 == 0);
}

TEST(komshina_d_grid_torus_mpi, check_target_in_range) {
  boost::mpi::communicator world;
  if (world.size() < 4) {
    GTEST_SKIP();
  }

  komshina_d_grid_torus_mpi::TestTaskMPI::TaskData input_data("sample_task", 2);

  ASSERT_TRUE(input_data.target >= 0 && input_data.target < world.size());
}

TEST(komshina_d_grid_torus_mpi, test_task_validation_and_preprocessing) {
  boost::mpi::communicator world;
  if (world.size() < 4) {
    GTEST_SKIP();
  }

  komshina_d_grid_torus_mpi::TestTaskMPI::TaskData input_data("sample_task", 2);
  komshina_d_grid_torus_mpi::TestTaskMPI::TaskData output_data;

  std::shared_ptr<ppc::core::TaskData> task_data_mpi = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    task_data_mpi->inputs.emplace_back(reinterpret_cast<uint8_t*>(&input_data));
    task_data_mpi->inputs_count.emplace_back(1);

    task_data_mpi->outputs.emplace_back(reinterpret_cast<uint8_t*>(&output_data));
    task_data_mpi->outputs_count.emplace_back(1);
  }

  komshina_d_grid_torus_mpi::TestTaskMPI task(task_data_mpi);

  ASSERT_TRUE(task.ValidationImpl());
  ASSERT_TRUE(task.PreProcessingImpl());
}

TEST(komshina_d_grid_torus_mpi, run_task_and_check_results) {
  boost::mpi::communicator world;
  if (world.size() < 4) {
    GTEST_SKIP();
  }

  komshina_d_grid_torus_mpi::TestTaskMPI::TaskData input_data("sample_task", 2);
  komshina_d_grid_torus_mpi::TestTaskMPI::TaskData output_data;

  std::shared_ptr<ppc::core::TaskData> task_data_mpi = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    task_data_mpi->inputs.emplace_back(reinterpret_cast<uint8_t*>(&input_data));
    task_data_mpi->inputs_count.emplace_back(1);

    task_data_mpi->outputs.emplace_back(reinterpret_cast<uint8_t*>(&output_data));
    task_data_mpi->outputs_count.emplace_back(1);
  }

  komshina_d_grid_torus_mpi::TestTaskMPI task(task_data_mpi);

  task.RunImpl();
  task.PostProcessingImpl();

  if (world.rank() == 0) {
    ASSERT_EQ(output_data.payload, input_data.payload);
    ASSERT_EQ(output_data.path, input_data.path);
  }
}

TEST(komshina_d_grid_torus_mpi, check_task_invalid_if_world_size_not_even) {
  boost::mpi::communicator world;
  if (world.size() < 4) {
    GTEST_SKIP();
  }

  komshina_d_grid_torus_mpi::TestTaskMPI::TaskData input_data("sample_task", 2);
  komshina_d_grid_torus_mpi::TestTaskMPI::TaskData output_data;

  std::shared_ptr<ppc::core::TaskData> task_data_mpi = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    task_data_mpi->inputs.emplace_back(reinterpret_cast<uint8_t*>(&input_data));
    task_data_mpi->inputs_count.emplace_back(1);

    task_data_mpi->outputs.emplace_back(reinterpret_cast<uint8_t*>(&output_data));
    task_data_mpi->outputs_count.emplace_back(1);
  }

  komshina_d_grid_torus_mpi::TestTaskMPI task(task_data_mpi);

  if (world.size() % 2 != 0) {
    ASSERT_FALSE(task.ValidationImpl());
  }
}

TEST(komshina_d_grid_torus_mpi, target_out_of_range) {
  boost::mpi::communicator world;
  if (world.size() < 4) {
    GTEST_SKIP();
  }

  komshina_d_grid_torus_mpi::TestTaskMPI::TaskData input_data("test_target", 1000000);
  komshina_d_grid_torus_mpi::TestTaskMPI::TaskData output_data;

  std::shared_ptr<ppc::core::TaskData> task_data_mpi = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    task_data_mpi->inputs.emplace_back(reinterpret_cast<uint8_t*>(&input_data));
    task_data_mpi->inputs_count.emplace_back(1);

    task_data_mpi->outputs.emplace_back(reinterpret_cast<uint8_t*>(&output_data));
    task_data_mpi->outputs_count.emplace_back(1);
  }

  komshina_d_grid_torus_mpi::TestTaskMPI task(task_data_mpi);

  ASSERT_TRUE(task.ValidationImpl());
  if (world.rank() == 0) {
    ASSERT_FALSE(task.PreProcessingImpl());
  }
}

TEST(komshina_d_grid_torus_mpi, normal_input_with_custom_data) {
  boost::mpi::communicator world;
  if (world.size() < 4) {
    GTEST_SKIP();
  }

  komshina_d_grid_torus_mpi::TestTaskMPI::TaskData input_data("input_data_v2", 4);
  komshina_d_grid_torus_mpi::TestTaskMPI::TaskData output_data;

  std::vector<int> expected_path = {0, 2, 4};
  std::shared_ptr<ppc::core::TaskData> task_data_mpi = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    task_data_mpi->inputs.emplace_back(reinterpret_cast<uint8_t*>(&input_data));
    task_data_mpi->inputs_count.emplace_back(1);

    task_data_mpi->outputs.emplace_back(reinterpret_cast<uint8_t*>(&output_data));
    task_data_mpi->outputs_count.emplace_back(1);
  }

  komshina_d_grid_torus_mpi::TestTaskMPI task(task_data_mpi);

  ASSERT_TRUE(task.ValidationImpl());
  ASSERT_TRUE(task.PreProcessingImpl());

  task.RunImpl();
  task.PostProcessingImpl();

  if (world.rank() == 0) {
    ASSERT_EQ(output_data.payload, input_data.payload);
    ASSERT_EQ(output_data.path, expected_path);
  }
}

TEST(komshina_d_grid_torus_mpi, process_large_input_string) {
  boost::mpi::communicator world;
  if (world.size() < 4) {
    GTEST_SKIP();
  }

  std::string large_input(500'000, 'b');
  komshina_d_grid_torus_mpi::TestTaskMPI::TaskData input_data(large_input, 5);
  komshina_d_grid_torus_mpi::TestTaskMPI::TaskData output_data;

  std::vector<int> expected_path = {0, 2, 5};
  std::shared_ptr<ppc::core::TaskData> task_data_mpi = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    task_data_mpi->inputs.emplace_back(reinterpret_cast<uint8_t*>(&input_data));
    task_data_mpi->inputs_count.emplace_back(1);

    task_data_mpi->outputs.emplace_back(reinterpret_cast<uint8_t*>(&output_data));
    task_data_mpi->outputs_count.emplace_back(1);
  }

  komshina_d_grid_torus_mpi::TestTaskMPI task(task_data_mpi);

  ASSERT_TRUE(task.ValidationImpl());
  ASSERT_TRUE(task.PreProcessingImpl());

  task.RunImpl();
  task.PostProcessingImpl();

  if (world.rank() == 0) {
    ASSERT_EQ(output_data.payload, input_data.payload);
    ASSERT_EQ(output_data.path, expected_path);
  }
}

TEST(komshina_d_grid_torus_mpi, dynamic_processor_count_test) {
  boost::mpi::communicator world;
  if (world.size() < 4) {
    GTEST_SKIP();
  }

  int target = world.size() / 4 * 3;
  std::vector<int> expected_path = komshina_d_grid_torus_mpi::TestTaskMPI::ComputePath(target, world.size(), 4, 4);

  komshina_d_grid_torus_mpi::TestTaskMPI::TaskData input_data("dynamic_data", target);
  komshina_d_grid_torus_mpi::TestTaskMPI::TaskData output_data;

  std::shared_ptr<ppc::core::TaskData> task_data_mpi = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    task_data_mpi->inputs.emplace_back(reinterpret_cast<uint8_t*>(&input_data));
    task_data_mpi->inputs_count.emplace_back(1);

    task_data_mpi->outputs.emplace_back(reinterpret_cast<uint8_t*>(&output_data));
    task_data_mpi->outputs_count.emplace_back(1);
  }

  komshina_d_grid_torus_mpi::TestTaskMPI task(task_data_mpi);

  ASSERT_TRUE(task.ValidationImpl());
  ASSERT_TRUE(task.PreProcessingImpl());

  task.RunImpl();
  task.PostProcessingImpl();

  if (world.rank() == 0) {
    ASSERT_EQ(output_data.payload, input_data.payload);
    ASSERT_EQ(output_data.path, expected_path);
  }
}