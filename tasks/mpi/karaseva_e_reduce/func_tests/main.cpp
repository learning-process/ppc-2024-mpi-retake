#include <gtest/gtest.h>
#include <mpi.h>

#include <numeric>
#include <vector>

#include "mpi/karaseva_e_reduce/include/ops_mpi.hpp"

TEST(karaseva_e_reduce_mpi, test_reduce_large_matrix) {
  MPI_Comm world = MPI_COMM_WORLD;
  int rank, size;
  MPI_Comm_rank(world, &rank);
  MPI_Comm_size(world, &size);

  constexpr size_t N = 1000;                 
  std::vector<int> local_data(N, rank + 1);  
  std::vector<int> reduced_data(N, 0);

  // Создаем task_data
  auto task_data_mpi = std::make_shared<ppc::core::TaskData>();
  task_data_mpi->inputs.emplace_back(reinterpret_cast<uint8_t *>(local_data.data()));
  task_data_mpi->inputs_count.emplace_back(local_data.size());
  task_data_mpi->outputs.emplace_back(reinterpret_cast<uint8_t *>(reduced_data.data()));
  task_data_mpi->outputs_count.emplace_back(reduced_data.size());

  // Запускаем кастомный Reduce
  karaseva_e_reduce_mpi::TestTaskMPI test_task_mpi(task_data_mpi, local_data.size());
  ASSERT_TRUE(test_task_mpi.Validation());
  test_task_mpi.PreProcessing();
  test_task_mpi.Run();
  test_task_mpi.PostProcessing();

  // Проверяем результат
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
  int rank, size;
  MPI_Comm_rank(world, &rank);
  MPI_Comm_size(world, &size);

  constexpr size_t N = 1000;
  std::vector<int> local_data(N, rank + 1);
  std::vector<int> custom_reduce_result(N, 0);
  std::vector<int> mpi_reduce_result(N, 0);

  // Создаем task_data
  auto task_data_mpi = std::make_shared<ppc::core::TaskData>();
  task_data_mpi->inputs.emplace_back(reinterpret_cast<uint8_t *>(local_data.data()));
  task_data_mpi->inputs_count.emplace_back(local_data.size());
  task_data_mpi->outputs.emplace_back(reinterpret_cast<uint8_t *>(custom_reduce_result.data()));
  task_data_mpi->outputs_count.emplace_back(custom_reduce_result.size());

  // Запускаем кастомный Reduce
  karaseva_e_reduce_mpi::TestTaskMPI test_task_mpi(task_data_mpi, local_data.size());
  ASSERT_TRUE(test_task_mpi.Validation());
  test_task_mpi.PreProcessing();
  test_task_mpi.Run();
  test_task_mpi.PostProcessing();

  // Запускаем стандартный MPI_Reduce
  MPI_Reduce(local_data.data(), mpi_reduce_result.data(), N, MPI_INT, MPI_SUM, 0, world);

  // Проверяем результат
  if (rank == 0) {
    EXPECT_EQ(custom_reduce_result, mpi_reduce_result);
  }
}


//
//#include <gtest/gtest.h>
//
//#include <cstddef>
//#include <cstdint>
//#include <fstream>
//#include <memory>
//#include <string>
//#include <vector>
//
//#include "core/task/include/task.hpp"
//#include "core/util/include/util.hpp"
//#include "mpi/karaseva_e_reduce/include/ops_mpi.hpp"
//
//TEST(karaseva_e_reduce_mpi, test_matmul_50) {
//  constexpr size_t kCount = 50;
//
//  // Create data
//  std::vector<int> in(kCount * kCount, 0);
//  std::vector<int> out(kCount * kCount, 0);
//
//  for (size_t i = 0; i < kCount; i++) {
//    in[(i * kCount) + i] = 1;
//  }
//
//  // Create task_data
//  auto task_data_mpi = std::make_shared<ppc::core::TaskData>();
//  task_data_mpi->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
//  task_data_mpi->inputs_count.emplace_back(in.size());
//  task_data_mpi->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
//  task_data_mpi->outputs_count.emplace_back(out.size());
//
//  // Create Task
//  karaseva_e_reduce_mpi::TestTaskMPI test_task_mpi(task_data_mpi);
//  ASSERT_EQ(test_task_mpi.Validation(), true);
//  test_task_mpi.PreProcessing();
//  test_task_mpi.Run();
//  test_task_mpi.PostProcessing();
//
//  EXPECT_EQ(in, out);
//}
//
//TEST(karaseva_e_reduce_mpi, test_matmul_100_from_file) {
//  std::string line;
//  std::ifstream test_file(ppc::util::GetAbsolutePath("mpi/karaseva_e_reduce/data/test.txt"));
//  if (test_file.is_open()) {
//    getline(test_file, line);
//  }
//  test_file.close();
//
//  const size_t count = std::stoi(line);
//
//  // Create data
//  std::vector<int> in(count * count, 0);
//  std::vector<int> out(count * count, 0);
//
//  for (size_t i = 0; i < count; i++) {
//    in[(i * count) + i] = 1;
//  }
//
//  // Create task_data
//  auto task_data_mpi = std::make_shared<ppc::core::TaskData>();
//  task_data_mpi->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
//  task_data_mpi->inputs_count.emplace_back(in.size());
//  task_data_mpi->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
//  task_data_mpi->outputs_count.emplace_back(out.size());
//
//  // Create Task
//  karaseva_e_reduce_mpi::TestTaskMPI test_task_mpi(task_data_mpi);
//  ASSERT_EQ(test_task_mpi.Validation(), true);
//  test_task_mpi.PreProcessing();
//  test_task_mpi.Run();
//  test_task_mpi.PostProcessing();
//
//  EXPECT_EQ(in, out);
//}
