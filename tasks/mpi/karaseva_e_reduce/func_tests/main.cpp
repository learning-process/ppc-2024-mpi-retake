#include <gtest/gtest.h>

#include <boost/mpi/communicator.hpp>
#include <boost/mpi/environment.hpp>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <numeric>
#include <random>
#include <vector>

#include "core/task/include/task.hpp"
#include "mpi/karaseva_e_reduce/include/ops_mpi.hpp"

namespace {

// Utility function to generate random data
template <typename T>
static std::vector<T> getRandom(int size) {
  std::random_device dev;
  std::mt19937 gen(dev());
  std::uniform_int_distribution<> distrib(-100, 100);
  std::vector<T> vec(size);
  for (int i = 0; i < size; i++) {
    vec[i] = static_cast<T>(distrib(gen));
  }
  return vec;
}

}  // namespace

// MPI_INT
TEST(karaseva_e_reduce_mpi, test_reduce_int) {
  constexpr size_t kCount = 50;

  std::vector<int> in = getRandom<int>(kCount);
  std::vector<int> out(1, 0);

  auto task_data_mpi = std::make_shared<ppc::core::TaskData>();
  task_data_mpi->inputs.emplace_back(reinterpret_cast<uint8_t*>(in.data()));
  task_data_mpi->inputs_count.emplace_back(in.size());
  task_data_mpi->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
  task_data_mpi->outputs_count.emplace_back(out.size());

  // Create the task with MPI-specific operations
  karaseva_e_reduce_mpi::TestTaskMPI<int> test_task_mpi(task_data_mpi);

  // Validate inputs
  ASSERT_TRUE(test_task_mpi.ValidationImpl());

  // Run the full pipeline
  test_task_mpi.PreProcessingImpl();
  test_task_mpi.RunImpl();
  test_task_mpi.PostProcessingImpl();

  // Calculate the expected result (sum of input vector)
  int expected_result = std::accumulate(in.begin(), in.end(), 0);

  // Check if the result matches the expected value
  EXPECT_EQ(out[0], expected_result);
}

// MPI_DOUBLE
TEST(karaseva_e_reduce_mpi, test_reduce_double) {
  constexpr size_t kCount = 50;

  std::vector<double> in = getRandom<double>(kCount);
  std::vector<double> out(1, 0.0);

  auto task_data_mpi = std::make_shared<ppc::core::TaskData>();
  task_data_mpi->inputs.emplace_back(reinterpret_cast<uint8_t*>(in.data()));
  task_data_mpi->inputs_count.emplace_back(in.size());
  task_data_mpi->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
  task_data_mpi->outputs_count.emplace_back(out.size());

  // Create the task with MPI-specific operations
  karaseva_e_reduce_mpi::TestTaskMPI<double> test_task_mpi(task_data_mpi);

  // Validate inputs
  ASSERT_TRUE(test_task_mpi.ValidationImpl());

  // Run the full pipeline
  test_task_mpi.PreProcessingImpl();
  test_task_mpi.RunImpl();
  test_task_mpi.PostProcessingImpl();

  // Calculate the expected result (sum of input vector)
  double expected_result = std::accumulate(in.begin(), in.end(), 0.0);

  // Check if the result matches the expected value
  EXPECT_DOUBLE_EQ(out[0], expected_result);
}

// MPI_FLOAT
TEST(karaseva_e_reduce_mpi, test_reduce_float) {
  constexpr size_t kCount = 50;

  std::vector<float> in = getRandom<float>(kCount);
  std::vector<float> out(1, 0.0F);

  auto task_data_mpi = std::make_shared<ppc::core::TaskData>();
  task_data_mpi->inputs.emplace_back(reinterpret_cast<uint8_t*>(in.data()));
  task_data_mpi->inputs_count.emplace_back(in.size());
  task_data_mpi->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
  task_data_mpi->outputs_count.emplace_back(out.size());

  // Create the task with MPI-specific operations
  karaseva_e_reduce_mpi::TestTaskMPI<float> test_task_mpi(task_data_mpi);

  // Validate inputs
  ASSERT_TRUE(test_task_mpi.ValidationImpl());

  // Run the full pipeline
  test_task_mpi.PreProcessingImpl();
  test_task_mpi.RunImpl();
  test_task_mpi.PostProcessingImpl();

  // Calculate the expected result (sum of input vector)
  float expected_result = std::accumulate(in.begin(), in.end(), 0.0F);

  // Check if the result matches the expected value
  EXPECT_FLOAT_EQ(out[0], expected_result);
}
