#include <gtest/gtest.h>
#include <mpi.h>

<<<<<<< HEAD
#include <chrono>
#include <fstream>
#include <iostream>
#include <vector>

#include "core/task/include/task.hpp"
#include "core/util/include/util.hpp"
#include "mpi/karaseva_e_reduce/include/ops_mpi.hpp"

// Тест с типом данных MPI_INT
TEST(karaseva_e_reduce_mpi, test_reduce_int) {
  constexpr size_t kCount = 50;

  // Create data
  std::vector<int> in(kCount, 1);  // Вектор с элементами типа int, все равны 1
  std::vector<int> out(1, 0);      // Один выходной элемент для хранения суммы

  // Create task_data
  auto task_data_mpi = std::make_shared<ppc::core::TaskData>();
  task_data_mpi->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_mpi->inputs_count.emplace_back(in.size());
  task_data_mpi->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_mpi->outputs_count.emplace_back(out.size());

  // Create Task
  karaseva_e_reduce_mpi::TestTaskMPI<int> test_task_mpi(task_data_mpi);

  // Замер времени выполнения
  auto start = std::chrono::high_resolution_clock::now();

  ASSERT_EQ(test_task_mpi.Validation(), true);
=======
#include <vector>

#include "mpi/karaseva_e_reduce/include/ops_mpi.hpp"

TEST(karaseva_e_reduce_mpi, test_reduce_large_matrix) {
  MPI_Comm world = MPI_COMM_WORLD;
  int rank = 0, size = 0;
  MPI_Comm_rank(world, &rank);
  MPI_Comm_size(world, &size);

  constexpr size_t N = 1000;
  std::vector<int> local_data(N, rank + 1);
  std::vector<int> reduced_data(N, 0);

  // Create task_data
  auto task_data_mpi = std::make_shared<ppc::core::TaskData>();
  task_data_mpi->inputs.emplace_back(reinterpret_cast<uint8_t *>(local_data.data()));
  task_data_mpi->inputs_count.emplace_back(local_data.size());
  task_data_mpi->outputs.emplace_back(reinterpret_cast<uint8_t *>(reduced_data.data()));
  task_data_mpi->outputs_count.emplace_back(reduced_data.size());

  // Reduce
  karaseva_e_reduce_mpi::TestTaskMPI test_task_mpi(task_data_mpi, local_data.size());
  ASSERT_TRUE(test_task_mpi.Validation());
>>>>>>> 8c0b0a4bb0e393c52cb48d47e5dccf68736a6c16
  test_task_mpi.PreProcessing();
  test_task_mpi.Run();
  test_task_mpi.PostProcessing();

<<<<<<< HEAD
  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> duration = end - start;
  std::cout << "Time taken for reduce operation on int vector: " << duration.count() << " seconds" << std::endl;

  // Для всех процессов сумма должна быть равна kCount
  EXPECT_EQ(out[0], kCount);
}

// Тест с типом данных MPI_DOUBLE
TEST(karaseva_e_reduce_mpi, test_reduce_double) {
  constexpr size_t kCount = 50;

  // Create data
  std::vector<double> in(kCount, 1.0);  // Вектор с элементами типа double, все равны 1.0
  std::vector<double> out(1, 0.0);      // Один выходной элемент для хранения суммы

  // Create task_data
  auto task_data_mpi = std::make_shared<ppc::core::TaskData>();
  task_data_mpi->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_mpi->inputs_count.emplace_back(in.size());
  task_data_mpi->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_mpi->outputs_count.emplace_back(out.size());

  // Create Task
  karaseva_e_reduce_mpi::TestTaskMPI<double> test_task_mpi(task_data_mpi);

  // Замер времени выполнения
  auto start = std::chrono::high_resolution_clock::now();

  ASSERT_EQ(test_task_mpi.Validation(), true);
=======
  // Checking the result
  if (rank == 0) {
    std::vector<int> expected_result(N, 0);
    for (int p = 0; p < size; ++p) {
      for (size_t i = 0; i < N; ++i) {
        expected_result[i] += (p + 1);
      }
    }
    EXPECT_EQ(reduced_data, expected_result);
  }
}

TEST(karaseva_e_reduce_mpi, test_reduce_vs_mpi_reduce) {
  MPI_Comm world = MPI_COMM_WORLD;
  int rank = 0, size = 0;
  MPI_Comm_rank(world, &rank);
  MPI_Comm_size(world, &size);

  constexpr size_t N = 1000;
  std::vector<int> local_data(N, rank + 1);
  std::vector<int> custom_reduce_result(N, 0);
  std::vector<int> mpi_reduce_result(N, 0);

  // Create task_data
  auto task_data_mpi = std::make_shared<ppc::core::TaskData>();
  task_data_mpi->inputs.emplace_back(reinterpret_cast<uint8_t *>(local_data.data()));
  task_data_mpi->inputs_count.emplace_back(local_data.size());
  task_data_mpi->outputs.emplace_back(reinterpret_cast<uint8_t *>(custom_reduce_result.data()));
  task_data_mpi->outputs_count.emplace_back(custom_reduce_result.size());

  // Reduce
  karaseva_e_reduce_mpi::TestTaskMPI test_task_mpi(task_data_mpi, local_data.size());
  ASSERT_TRUE(test_task_mpi.Validation());
>>>>>>> 8c0b0a4bb0e393c52cb48d47e5dccf68736a6c16
  test_task_mpi.PreProcessing();
  test_task_mpi.Run();
  test_task_mpi.PostProcessing();

<<<<<<< HEAD
  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> duration = end - start;
  std::cout << "Time taken for reduce operation on double vector: " << duration.count() << " seconds" << std::endl;

  // Для всех процессов сумма должна быть равна kCount
  EXPECT_EQ(out[0], kCount);
}

// Тест с типом данных MPI_FLOAT
TEST(karaseva_e_reduce_mpi, test_reduce_float) {
  constexpr size_t kCount = 50;

  // Create data
  std::vector<float> in(kCount, 1.0f);  // Вектор с элементами типа float, все равны 1.0f
  std::vector<float> out(1, 0.0f);      // Один выходной элемент для хранения суммы

  // Create task_data
  auto task_data_mpi = std::make_shared<ppc::core::TaskData>();
  task_data_mpi->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_mpi->inputs_count.emplace_back(in.size());
  task_data_mpi->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_mpi->outputs_count.emplace_back(out.size());

  // Create Task
  karaseva_e_reduce_mpi::TestTaskMPI<float> test_task_mpi(task_data_mpi);

  // Замер времени выполнения
  auto start = std::chrono::high_resolution_clock::now();

  ASSERT_EQ(test_task_mpi.Validation(), true);
  test_task_mpi.PreProcessing();
  test_task_mpi.Run();
  test_task_mpi.PostProcessing();

  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> duration = end - start;
  std::cout << "Time taken for reduce operation on float vector: " << duration.count() << " seconds" << std::endl;

  // Для всех процессов сумма должна быть равна kCount
  EXPECT_EQ(out[0], kCount);
=======
  // MPI_Reduce
  MPI_Reduce(local_data.data(), mpi_reduce_result.data(), N, MPI_INT, MPI_SUM, 0, world);

  // Checking the result
  if (rank == 0) {
    EXPECT_EQ(custom_reduce_result, mpi_reduce_result);
  }
>>>>>>> 8c0b0a4bb0e393c52cb48d47e5dccf68736a6c16
}