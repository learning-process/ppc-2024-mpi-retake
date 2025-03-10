#include <gtest/gtest.h>

#include <algorithm>
#include <boost/mpi/collectives/broadcast.hpp>
#include <boost/mpi/communicator.hpp>
#include <boost/serialization/vector.hpp>  //NOLINT
#include <cstdint>
#include <limits>
#include <memory>
#include <random>
#include <vector>

#include "core/task/include/task.hpp"
#include "mpi/vinyaikina_e_max_of_vector_elements/include/ops_mpi.hpp"

namespace {
std::vector<int32_t> MakeRandomVector(int32_t size, int32_t val_min, int32_t val_max) {
  std::random_device rd;
  std::mt19937 gen(static_cast<int>(rd()));
  std::uniform_int_distribution<> distrib(val_min, val_max);

  std::vector<int32_t> new_vector(size);
  std::ranges::generate(new_vector.begin(), new_vector.end(), [&]() { return distrib(gen); });
  return new_vector;
}

void RunParallelAndSequentialTasks(std::vector<int32_t>& input_vector, int32_t expected_max) {
  boost::mpi::communicator world;
  int32_t result_parallel = std::numeric_limits<int32_t>::min();

  auto task_data_par = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    task_data_par->inputs.emplace_back(reinterpret_cast<uint8_t*>(input_vector.data()));
    task_data_par->inputs_count.emplace_back(input_vector.size());
    task_data_par->outputs.emplace_back(reinterpret_cast<uint8_t*>(&result_parallel));
    task_data_par->outputs_count.emplace_back(1);
  }

  vinyaikina_e_max_of_vector_elements::VectorMaxPar test_mpi_task_parallel(task_data_par);
  test_mpi_task_parallel.ValidationImpl();
  test_mpi_task_parallel.PreProcessingImpl();
  test_mpi_task_parallel.RunImpl();
  test_mpi_task_parallel.PostProcessingImpl();

  if (world.rank() == 0) {
    ASSERT_EQ(result_parallel, expected_max);
  }
}
}  // namespace

TEST(vinyaikina_e_max_of_vector_elements, randomVector50000) {
  boost::mpi::communicator world;
  std::vector<int32_t> input_vector;

  if (world.rank() == 0) {
    input_vector = MakeRandomVector(50000, -500, 5000);
  }

  boost::mpi::broadcast(world, input_vector, 0);

  int32_t expected_max = 0;
  expected_max = std::numeric_limits<int32_t>::min();
  if (world.rank() == 0) {
    expected_max = *std::ranges::max_element(input_vector.begin(), input_vector.end());
  }

  RunParallelAndSequentialTasks(input_vector, expected_max);
}

TEST(vinyaikina_e_max_of_vector_elements, regularVector) {
  std::vector<int32_t> input_vector = {1, 2, 3, -5, 3, 43};
  RunParallelAndSequentialTasks(input_vector, 43);
}

TEST(vinyaikina_e_max_of_vector_elements, positiveNumbers) {
  std::vector<int32_t> input_vector = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
  RunParallelAndSequentialTasks(input_vector, 10);
}

TEST(vinyaikina_e_max_of_vector_elements, negativeNumbers) {
  std::vector<int32_t> input_vector = {-1, -2, -3, -4, -5, -6, -7, -8, -9, -10};
  RunParallelAndSequentialTasks(input_vector, -1);
}

TEST(vinyaikina_e_max_of_vector_elements, zeroVector) {
  std::vector<int32_t> input_vector = {0, 0, 0, 0, 0};
  RunParallelAndSequentialTasks(input_vector, 0);
}

TEST(vinyaikina_e_max_of_vector_elements, tinyVector) {
  std::vector<int32_t> input_vector = {4, -20};
  RunParallelAndSequentialTasks(input_vector, 4);
}

TEST(vinyaikina_e_max_of_vector_elements, emptyVector) {
  std::vector<int32_t> input_vector = {};
  RunParallelAndSequentialTasks(input_vector, std::numeric_limits<int32_t>::min());
}

TEST(vinyaikina_e_max_of_vector_elements, validationNotPassed) {
  boost::mpi::communicator world;
  std::vector<int32_t> input = {1, 2, 3, -5};
  auto task_data = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    task_data->inputs_count.emplace_back(input.size());
    task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(input.data()));
  }

  vinyaikina_e_max_of_vector_elements::VectorMaxPar vector_max_par(task_data);

  if (world.rank() == 0) {
    ASSERT_FALSE(vector_max_par.ValidationImpl());
  }
}