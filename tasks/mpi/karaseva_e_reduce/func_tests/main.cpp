#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

#include "core/task/include/task.hpp"
#include "mpi/karaseva_e_reduce/include/ops_mpi.hpp"

// MPI_INT
TEST(karaseva_e_reduce_mpi, test_reduce_int) {
  constexpr size_t kCount = 50;

  std::vector<int> in(kCount, 1);
  std::vector<int> out(1, 0);

  auto task_data_mpi = std::make_shared<ppc::core::TaskData>();
  task_data_mpi->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_mpi->inputs_count.emplace_back(in.size());
  task_data_mpi->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_mpi->outputs_count.emplace_back(out.size());

  karaseva_e_reduce_mpi::TestTaskMPI<int> test_task_mpi(task_data_mpi);

  ASSERT_EQ(test_task_mpi.Validation(), true);
  test_task_mpi.PreProcessing();
  test_task_mpi.Run();
  test_task_mpi.PostProcessing();

  EXPECT_EQ(out[0], static_cast<int>(kCount));
}

// MPI_DOUBLE
TEST(karaseva_e_reduce_mpi, test_reduce_double) {
  constexpr size_t kCount = 50;

  std::vector<double> in(kCount, 1.0);
  std::vector<double> out(1, 0.0);

  auto task_data_mpi = std::make_shared<ppc::core::TaskData>();
  task_data_mpi->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_mpi->inputs_count.emplace_back(in.size());
  task_data_mpi->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_mpi->outputs_count.emplace_back(out.size());

  karaseva_e_reduce_mpi::TestTaskMPI<double> test_task_mpi(task_data_mpi);

  ASSERT_EQ(test_task_mpi.Validation(), true);
  test_task_mpi.PreProcessing();
  test_task_mpi.Run();
  test_task_mpi.PostProcessing();

  EXPECT_DOUBLE_EQ(out[0], static_cast<double>(kCount));
}

// MPI_FLOAT
TEST(karaseva_e_reduce_mpi, test_reduce_float) {
  constexpr size_t kCount = 50;

  std::vector<float> in(kCount, 1.0F);
  std::vector<float> out(1, 0.0F);

  auto task_data_mpi = std::make_shared<ppc::core::TaskData>();
  task_data_mpi->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_mpi->inputs_count.emplace_back(in.size());
  task_data_mpi->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_mpi->outputs_count.emplace_back(out.size());

  karaseva_e_reduce_mpi::TestTaskMPI<float> test_task_mpi(task_data_mpi);

  ASSERT_EQ(test_task_mpi.Validation(), true);
  test_task_mpi.PreProcessing();
  test_task_mpi.Run();
  test_task_mpi.PostProcessing();

  EXPECT_FLOAT_EQ(out[0], static_cast<float>(kCount));
}