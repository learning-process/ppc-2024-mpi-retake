#define OMPI_SKIP_MPICXX

#include <gtest/gtest.h>
#include <mpi.h>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <vector>

#include "core/task/include/task.hpp"
#include "mpi/muradov_k_odd_even_batcher_sort/include/ops_mpi.hpp"

TEST(muradov_k_odd_even_batcher_sort_func, positive_values) {
  int proc_rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &proc_rank);

  std::vector<int> input = {9, 8, 7, 1, 5};
  std::vector<int> expected = {1, 5, 7, 8, 9};
  std::vector<int> output(input.size(), 0);

  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(input.data()));
  task_data->inputs_count.emplace_back(input.size());
  task_data->outputs.emplace_back(reinterpret_cast<uint8_t*>(output.data()));
  task_data->outputs_count.emplace_back(output.size());
  task_data->state_of_testing = ppc::core::TaskData::kFunc;

  muradov_k_odd_even_batcher_sort::OddEvenBatcherSortTask sort_task(task_data);
  ASSERT_TRUE(sort_task.Validation());
  sort_task.PreProcessing();
  sort_task.Run();
  sort_task.PostProcessing();

  if (proc_rank == 0) {
    ASSERT_EQ(output, expected);
  }
}

TEST(muradov_k_odd_even_batcher_sort_func, negative_values) {
  int proc_rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &proc_rank);

  std::vector<int> input = {-7, -8, -9, -1, -5};
  std::vector<int> expected = {-9, -8, -7, -5, -1};
  std::vector<int> output(input.size(), 0);

  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(input.data()));
  task_data->inputs_count.emplace_back(input.size());
  task_data->outputs.emplace_back(reinterpret_cast<uint8_t*>(output.data()));
  task_data->outputs_count.emplace_back(output.size());
  task_data->state_of_testing = ppc::core::TaskData::kFunc;

  muradov_k_odd_even_batcher_sort::OddEvenBatcherSortTask sort_task(task_data);
  ASSERT_TRUE(sort_task.Validation());
  sort_task.PreProcessing();
  sort_task.Run();
  sort_task.PostProcessing();

  if (proc_rank == 0) {
    ASSERT_EQ(output, expected);
  }
}

TEST(muradov_k_odd_even_batcher_sort_func, mixed_values) {
  int proc_rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &proc_rank);

  std::vector<int> input = {-7, 8, 0, -1, 5};
  std::vector<int> expected = {-7, -1, 0, 5, 8};
  std::vector<int> output(input.size(), 0);

  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(input.data()));
  task_data->inputs_count.emplace_back(input.size());
  task_data->outputs.emplace_back(reinterpret_cast<uint8_t*>(output.data()));
  task_data->outputs_count.emplace_back(output.size());
  task_data->state_of_testing = ppc::core::TaskData::kFunc;

  muradov_k_odd_even_batcher_sort::OddEvenBatcherSortTask sort_task(task_data);
  ASSERT_TRUE(sort_task.Validation());
  sort_task.PreProcessing();
  sort_task.Run();
  sort_task.PostProcessing();

  if (proc_rank == 0) {
    ASSERT_EQ(output, expected);
  }
}

TEST(muradov_k_odd_even_batcher_sort_func, compare_with_qsort) {
  int proc_rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &proc_rank);

  const int n = 235;
  std::vector<int> input(n);
  std::srand(static_cast<unsigned int>(std::time(nullptr)));
  for (int i = 0; i < n; ++i) {
    input[i] = std::rand() % 1000;
  }
  std::vector<int> expected = input;
  muradov_k_odd_even_batcher_sort::QSort(expected, 0, static_cast<int>(expected.size()) - 1);
  std::vector<int> output(input.size(), 0);

  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(input.data()));
  task_data->inputs_count.emplace_back(input.size());
  task_data->outputs.emplace_back(reinterpret_cast<uint8_t*>(output.data()));
  task_data->outputs_count.emplace_back(output.size());
  task_data->state_of_testing = ppc::core::TaskData::kFunc;

  muradov_k_odd_even_batcher_sort::OddEvenBatcherSortTask sort_task(task_data);
  ASSERT_TRUE(sort_task.Validation());
  sort_task.PreProcessing();
  sort_task.Run();
  sort_task.PostProcessing();

  if (proc_rank == 0) {
    ASSERT_EQ(output, expected);
  }
}

TEST(muradov_k_odd_even_batcher_sort_func, compare_with_std_sort) {
  int proc_rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &proc_rank);

  const int n = 235;
  std::vector<int> input(n);
  std::srand(static_cast<unsigned int>(std::time(nullptr)));
  for (int i = 0; i < n; ++i) {
    input[i] = std::rand() % 1000;
  }
  std::vector<int> expected = input;
  std::sort(expected.begin(), expected.end());
  std::vector<int> output(input.size(), 0);

  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(input.data()));
  task_data->inputs_count.emplace_back(input.size());
  task_data->outputs.emplace_back(reinterpret_cast<uint8_t*>(output.data()));
  task_data->outputs_count.emplace_back(output.size());
  task_data->state_of_testing = ppc::core::TaskData::kFunc;

  muradov_k_odd_even_batcher_sort::OddEvenBatcherSortTask sort_task(task_data);
  ASSERT_TRUE(sort_task.Validation());
  sort_task.PreProcessing();
  sort_task.Run();
  sort_task.PostProcessing();

  if (proc_rank == 0) {
    ASSERT_EQ(output, expected);
  }
}