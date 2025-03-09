#include <gtest/gtest.h>

#include <algorithm>
#include <boost/mpi/communicator.hpp>
#include <cstdint>
#include <memory>
#include <vector>

#include "core/task/include/task.hpp"
#include "mpi/Shpynov_N_radix_sort/include/Shpynov_N_radix_sort_mpi.hpp"

TEST(shpynov_n_radix_sort_mpi, test_single_num) {
  boost::mpi::communicator world;
  std::vector<int> input_vec(1, 0);

  std::vector<int> expected_result(1, 0);
  std::vector<int> returned_result(1, 0);
  std::shared_ptr<ppc::core::TaskData> task_data_mpi = std::make_shared<ppc::core::TaskData>();

  task_data_mpi->inputs.emplace_back(reinterpret_cast<uint8_t *>(input_vec.data()));
  task_data_mpi->inputs_count.emplace_back(1);
  task_data_mpi->outputs.emplace_back(reinterpret_cast<uint8_t *>(returned_result.data()));
  task_data_mpi->outputs_count.emplace_back(1);

  shpynov_n_radix_sort_mpi::TestTaskMPI test_task_mpi(task_data_mpi);
  if (world.rank() == 0) {
    ASSERT_EQ(test_task_mpi.ValidationImpl(), true);
  }
  test_task_mpi.PreProcessingImpl();
  test_task_mpi.RunImpl();
  test_task_mpi.PostProcessingImpl();

  if (world.rank() == 0) {
    ASSERT_EQ(expected_result, returned_result);
  }
}

TEST(shpynov_n_radix_sort_mpi, test_tiny_vector) {
  boost::mpi::communicator world;
  std::vector<int> input_vec = {33, 22};

  std::vector<int> expected_result = {22, 33};
  std::vector<int> returned_result(input_vec.size());
  std::shared_ptr<ppc::core::TaskData> task_data_mpi = std::make_shared<ppc::core::TaskData>();

  task_data_mpi->inputs.emplace_back(reinterpret_cast<uint8_t *>(input_vec.data()));
  task_data_mpi->inputs_count.emplace_back(input_vec.size());
  task_data_mpi->outputs.emplace_back(reinterpret_cast<uint8_t *>(returned_result.data()));
  task_data_mpi->outputs_count.emplace_back(returned_result.size());

  shpynov_n_radix_sort_mpi::TestTaskMPI test_task_mpi(task_data_mpi);
  if (world.rank() == 0) {
    ASSERT_EQ(test_task_mpi.ValidationImpl(), true);
  }
  test_task_mpi.PreProcessingImpl();
  test_task_mpi.RunImpl();
  test_task_mpi.PostProcessingImpl();

  if (world.rank() == 0) {
    ASSERT_EQ(expected_result, returned_result);
  }
}
TEST(shpynov_n_radix_sort_mpi, test_lots_of_zeros) {
  boost::mpi::communicator world;
  constexpr int kCount = 10;
  std::vector<int> input_vec = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
  std::vector<int> expected_result = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
  for (int i = 0; i < kCount; i++) {
    input_vec.insert(input_vec.end(), input_vec.begin(), input_vec.end());
    expected_result.insert(expected_result.end(), expected_result.begin(), expected_result.end());
  }
  std::vector<int> returned_result(input_vec.size());
  std::shared_ptr<ppc::core::TaskData> task_data_mpi = std::make_shared<ppc::core::TaskData>();

  task_data_mpi->inputs.emplace_back(reinterpret_cast<uint8_t *>(input_vec.data()));
  task_data_mpi->inputs_count.emplace_back(input_vec.size());
  task_data_mpi->outputs.emplace_back(reinterpret_cast<uint8_t *>(returned_result.data()));
  task_data_mpi->outputs_count.emplace_back(returned_result.size());

  shpynov_n_radix_sort_mpi::TestTaskMPI test_task_mpi(task_data_mpi);
  if (world.rank() == 0) {
    ASSERT_EQ(test_task_mpi.ValidationImpl(), true);
  }
  test_task_mpi.PreProcessingImpl();
  test_task_mpi.RunImpl();
  test_task_mpi.PostProcessingImpl();

  if (world.rank() == 0) {
    ASSERT_EQ(expected_result, returned_result);
  }
}
TEST(shpynov_n_radix_sort_mpi, test_some_numbers_diff_length) {
  boost::mpi::communicator world;
  std::vector<int> input_vec = {17, 33, 22, 420, 1, 0, 5837, 659};

  std::vector<int> expected_result = {0, 1, 17, 22, 33, 420, 659, 5837};
  std::vector<int> returned_result(input_vec.size());
  std::shared_ptr<ppc::core::TaskData> task_data_mpi = std::make_shared<ppc::core::TaskData>();

  task_data_mpi->inputs.emplace_back(reinterpret_cast<uint8_t *>(input_vec.data()));
  task_data_mpi->inputs_count.emplace_back(input_vec.size());
  task_data_mpi->outputs.emplace_back(reinterpret_cast<uint8_t *>(returned_result.data()));
  task_data_mpi->outputs_count.emplace_back(returned_result.size());

  shpynov_n_radix_sort_mpi::TestTaskMPI test_task_mpi(task_data_mpi);
  if (world.rank() == 0) {
    ASSERT_EQ(test_task_mpi.ValidationImpl(), true);
  }
  test_task_mpi.PreProcessingImpl();
  test_task_mpi.RunImpl();
  test_task_mpi.PostProcessingImpl();

  if (world.rank() == 0) {
    ASSERT_EQ(expected_result, returned_result);
  }
}

TEST(shpynov_n_radix_sort_mpi, test_some_numbers_diff_length_neg_numbers) {
  boost::mpi::communicator world;
  std::vector<int> input_vec = {-17, -33, -22, -420, -1, -5837, -659};

  std::vector<int> expected_result = {-5837, -659, -420, -33, -22, -17, -1};
  std::vector<int> returned_result(input_vec.size());
  std::shared_ptr<ppc::core::TaskData> task_data_mpi = std::make_shared<ppc::core::TaskData>();

  task_data_mpi->inputs.emplace_back(reinterpret_cast<uint8_t *>(input_vec.data()));
  task_data_mpi->inputs_count.emplace_back(input_vec.size());
  task_data_mpi->outputs.emplace_back(reinterpret_cast<uint8_t *>(returned_result.data()));
  task_data_mpi->outputs_count.emplace_back(returned_result.size());

  shpynov_n_radix_sort_mpi::TestTaskMPI test_task_mpi(task_data_mpi);
  if (world.rank() == 0) {
    ASSERT_EQ(test_task_mpi.ValidationImpl(), true);
  }
  test_task_mpi.PreProcessingImpl();
  test_task_mpi.RunImpl();
  test_task_mpi.PostProcessingImpl();

  if (world.rank() == 0) {
    ASSERT_EQ(expected_result, returned_result);
  }
}

TEST(shpynov_n_radix_sort_mpi, test_some_numbers_diff_length_pos_and_neg_numbers) {
  boost::mpi::communicator world;
  std::vector<int> input_vec = {17, 33, 22, 420, 1, 0, 5837, 659, -4, -28, -76, -110291};

  std::vector<int> expected_result = {-110291, -76, -28, -4, 0, 1, 17, 22, 33, 420, 659, 5837};
  std::vector<int> returned_result(input_vec.size());
  std::shared_ptr<ppc::core::TaskData> task_data_mpi = std::make_shared<ppc::core::TaskData>();

  task_data_mpi->inputs.emplace_back(reinterpret_cast<uint8_t *>(input_vec.data()));
  task_data_mpi->inputs_count.emplace_back(input_vec.size());
  task_data_mpi->outputs.emplace_back(reinterpret_cast<uint8_t *>(returned_result.data()));
  task_data_mpi->outputs_count.emplace_back(returned_result.size());

  shpynov_n_radix_sort_mpi::TestTaskMPI test_task_mpi(task_data_mpi);
  if (world.rank() == 0) {
    ASSERT_EQ(test_task_mpi.ValidationImpl(), true);
  }
  test_task_mpi.PreProcessingImpl();
  test_task_mpi.RunImpl();
  test_task_mpi.PostProcessingImpl();

  if (world.rank() == 0) {
    ASSERT_EQ(expected_result, returned_result);
  }
}

TEST(shpynov_n_radix_sort_mpi, test_some_numbers_diff_length_pos_and_neg_numbers_with_same_nums) {
  boost::mpi::communicator world;
  std::vector<int> input_vec = {17, 33, 22, 420, 1, 17, 0, 5837, 659, -4, -28, 0, -76, -4, -110291};

  std::vector<int> expected_result = {-110291, -76, -28, -4, -4, 0, 0, 1, 17, 17, 22, 33, 420, 659, 5837};
  std::vector<int> returned_result(input_vec.size());
  std::shared_ptr<ppc::core::TaskData> task_data_mpi = std::make_shared<ppc::core::TaskData>();

  task_data_mpi->inputs.emplace_back(reinterpret_cast<uint8_t *>(input_vec.data()));
  task_data_mpi->inputs_count.emplace_back(input_vec.size());
  task_data_mpi->outputs.emplace_back(reinterpret_cast<uint8_t *>(returned_result.data()));
  task_data_mpi->outputs_count.emplace_back(returned_result.size());

  shpynov_n_radix_sort_mpi::TestTaskMPI test_task_mpi(task_data_mpi);
  if (world.rank() == 0) {
    ASSERT_EQ(test_task_mpi.ValidationImpl(), true);
  }
  test_task_mpi.PreProcessingImpl();
  test_task_mpi.RunImpl();
  test_task_mpi.PostProcessingImpl();

  if (world.rank() == 0) {
    ASSERT_EQ(expected_result, returned_result);
  }
}

TEST(shpynov_n_radix_sort_mpi, test_invalid) {
  boost::mpi::communicator world;
  std::vector<int> input_vec;

  std::vector<int> expected_result = {-110291, -76, -28, -4, 0, 1, 17, 22, 33, 420, 659, 5837};
  std::vector<int> returned_result(input_vec.size());
  std::shared_ptr<ppc::core::TaskData> task_data_mpi = std::make_shared<ppc::core::TaskData>();

  task_data_mpi->inputs.emplace_back(reinterpret_cast<uint8_t *>(input_vec.data()));
  task_data_mpi->inputs_count.emplace_back(input_vec.size());
  task_data_mpi->outputs.emplace_back(reinterpret_cast<uint8_t *>(returned_result.data()));
  task_data_mpi->outputs_count.emplace_back(returned_result.size());

  shpynov_n_radix_sort_mpi::TestTaskMPI test_task_mpi(task_data_mpi);
  if (world.rank() == 0) {
    ASSERT_NE(test_task_mpi.ValidationImpl(), true);
  }
}

TEST(shpynov_n_radix_sort_mpi, test_tiny_random_vector) {
  boost::mpi::communicator world;
  std::vector<int> input_vec = shpynov_n_radix_sort_mpi::GetRandVec(2);
  std::vector<int> expected_result = input_vec;
  std::ranges::sort(expected_result.begin(), expected_result.end());

  std::vector<int> returned_result(input_vec.size());
  std::shared_ptr<ppc::core::TaskData> task_data_mpi = std::make_shared<ppc::core::TaskData>();

  task_data_mpi->inputs.emplace_back(reinterpret_cast<uint8_t *>(input_vec.data()));
  task_data_mpi->inputs_count.emplace_back(input_vec.size());
  task_data_mpi->outputs.emplace_back(reinterpret_cast<uint8_t *>(returned_result.data()));
  task_data_mpi->outputs_count.emplace_back(returned_result.size());

  shpynov_n_radix_sort_mpi::TestTaskMPI test_task_mpi(task_data_mpi);
  if (world.rank() == 0) {
    ASSERT_EQ(test_task_mpi.ValidationImpl(), true);
  }
  test_task_mpi.PreProcessingImpl();
  test_task_mpi.RunImpl();
  test_task_mpi.PostProcessingImpl();

  if (world.rank() == 0) {
    ASSERT_EQ(expected_result, returned_result);
  }
}

TEST(shpynov_n_radix_sort_mpi, test_average_random_vector) {
  boost::mpi::communicator world;
  std::vector<int> input_vec = shpynov_n_radix_sort_mpi::GetRandVec(30);
  std::vector<int> expected_result = input_vec;
  std::ranges::sort(expected_result.begin(), expected_result.end());

  std::vector<int> returned_result(input_vec.size());
  std::shared_ptr<ppc::core::TaskData> task_data_mpi = std::make_shared<ppc::core::TaskData>();

  task_data_mpi->inputs.emplace_back(reinterpret_cast<uint8_t *>(input_vec.data()));
  task_data_mpi->inputs_count.emplace_back(input_vec.size());
  task_data_mpi->outputs.emplace_back(reinterpret_cast<uint8_t *>(returned_result.data()));
  task_data_mpi->outputs_count.emplace_back(returned_result.size());

  shpynov_n_radix_sort_mpi::TestTaskMPI test_task_mpi(task_data_mpi);
  if (world.rank() == 0) {
    ASSERT_EQ(test_task_mpi.ValidationImpl(), true);
  }
  test_task_mpi.PreProcessingImpl();
  test_task_mpi.RunImpl();
  test_task_mpi.PostProcessingImpl();

  if (world.rank() == 0) {
    ASSERT_EQ(expected_result, returned_result);
  }
}

TEST(shpynov_n_radix_sort_mpi, test_big_random_vector) {
  boost::mpi::communicator world;
  std::vector<int> input_vec = shpynov_n_radix_sort_mpi::GetRandVec(2000);
  std::vector<int> expected_result = input_vec;
  std::ranges::sort(expected_result.begin(), expected_result.end());

  std::vector<int> returned_result(input_vec.size());
  std::shared_ptr<ppc::core::TaskData> task_data_mpi = std::make_shared<ppc::core::TaskData>();

  task_data_mpi->inputs.emplace_back(reinterpret_cast<uint8_t *>(input_vec.data()));
  task_data_mpi->inputs_count.emplace_back(input_vec.size());
  task_data_mpi->outputs.emplace_back(reinterpret_cast<uint8_t *>(returned_result.data()));
  task_data_mpi->outputs_count.emplace_back(returned_result.size());

  shpynov_n_radix_sort_mpi::TestTaskMPI test_task_mpi(task_data_mpi);
  if (world.rank() == 0) {
    ASSERT_EQ(test_task_mpi.ValidationImpl(), true);
  }
  test_task_mpi.PreProcessingImpl();
  test_task_mpi.RunImpl();
  test_task_mpi.PostProcessingImpl();

  if (world.rank() == 0) {
    ASSERT_EQ(expected_result, returned_result);
  }
}