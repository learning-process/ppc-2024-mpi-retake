#include <gtest/gtest.h>

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <memory>
#include <string>
#include <vector>

#include "core/task/include/task.hpp"
#include "core/util/include/util.hpp"
#include "mpi/shuravina_o_contrast/include/ops_mpi.hpp"

using namespace std::chrono;

TEST(shuravina_o_contrast, test_performance) {
  constexpr size_t kCount = 512;
  std::vector<uint8_t> in(kCount * kCount, 0);
  std::vector<uint8_t> out(kCount * kCount, 0);

  for (size_t i = 0; i < kCount; i++) {
    for (size_t j = 0; j < kCount; j++) {
      in[(i * kCount) + j] = static_cast<uint8_t>(i + j);
    }
  }

  auto task_data_mpi = std::make_shared<ppc::core::TaskData>();
  task_data_mpi->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_mpi->inputs_count.emplace_back(in.size());
  task_data_mpi->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_mpi->outputs_count.emplace_back(out.size());

  shuravina_o_contrast::TestTaskMPI test_task_mpi(task_data_mpi);

  auto start = high_resolution_clock::now();
  test_task_mpi.PreProcessing();
  test_task_mpi.Run();
  test_task_mpi.PostProcessing();
  auto end = high_resolution_clock::now();

  auto duration = duration_cast<milliseconds>(end - start);

  std::cout << "MPI version time: " << duration.count() << " ms" << std::endl;

  EXPECT_LT(duration.count(), 1000);
}
TEST(shuravina_o_contrast, test_performance_large_image) {
  constexpr size_t kCount = 1024;

  std::vector<uint8_t> in(kCount * kCount, 0);
  std::vector<uint8_t> out(kCount * kCount, 0);

  for (size_t i = 0; i < kCount; i++) {
    for (size_t j = 0; j < kCount; j++) {
      in[(i * kCount) + j] = static_cast<uint8_t>(i + j);
    }
  }

  auto task_data_mpi = std::make_shared<ppc::core::TaskData>();
  task_data_mpi->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_mpi->inputs_count.emplace_back(in.size());
  task_data_mpi->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_mpi->outputs_count.emplace_back(out.size());

  shuravina_o_contrast::TestTaskMPI test_task_mpi(task_data_mpi);

  auto start = high_resolution_clock::now();
  test_task_mpi.PreProcessing();
  test_task_mpi.Run();
  test_task_mpi.PostProcessing();
  auto end = high_resolution_clock::now();

  auto duration = duration_cast<milliseconds>(end - start);

  std::cout << "MPI version time (1024x1024): " << duration.count() << " ms" << std::endl;

  EXPECT_LT(duration.count(), 5000);
}