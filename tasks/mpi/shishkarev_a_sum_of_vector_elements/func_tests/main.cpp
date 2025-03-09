// Copyright 2023 Nesterov Alexander
#include <gtest/gtest.h>

#include <algorithm>
#include <boost/mpi/communicator.hpp>
#include <boost/mpi/environment.hpp>
#include <cstdint>
#include <memory>
#include <numeric>
#include <random>
#include <vector>

#include "core/task/include/task.hpp"
#include "mpi/shishkarev_a_sum_of_vector_elements/include/ops_mpi.hpp"

namespace {
std::vector<int> GetRandomVector(int vector_size) {
  std::mt19937 generator(std::random_device{}());
  std::uniform_int_distribution<int> distribution(0, 99);
  std::vector<int> random_vector(vector_size);
  std::ranges::generate(random_vector, [&]() { return distribution(generator); });
  return random_vector;
}
}  // namespace

TEST(shishkarev_a_sum_of_vector_elements_mpi, test_empty_sum) {
  boost::mpi::environment env;
  boost::mpi::communicator world;

  std::vector<int> global_vec;
  std::vector<int32_t> global_sum(1, 0);

  std::shared_ptr<ppc::core::TaskData> task_data_par = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    task_data_par->inputs.emplace_back(nullptr);
    task_data_par->inputs_count.emplace_back(0);
    task_data_par->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_sum.data()));
    task_data_par->outputs_count.emplace_back(global_sum.size());
  }

  shishkarev_a_sum_of_vector_elements_mpi::MPIVectorSumParallel parallel(task_data_par);
  ASSERT_TRUE(parallel.ValidationImpl());
  parallel.PreProcessingImpl();
  parallel.RunImpl();
  parallel.PostProcessingImpl();

  if (world.rank() == 0) {
    ASSERT_EQ(global_sum[0], 0);
  }
}

TEST(shishkarev_a_sum_of_vector_elements_mpi, test_single_element_sum) {
  boost::mpi::environment env;
  boost::mpi::communicator world;

  std::vector<int> global_vec(1, 42);
  std::vector<int32_t> global_sum(1, 0);

  std::shared_ptr<ppc::core::TaskData> task_data_par = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    task_data_par->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vec.data()));
    task_data_par->inputs_count.emplace_back(global_vec.size());
    task_data_par->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_sum.data()));
    task_data_par->outputs_count.emplace_back(global_sum.size());
  }

  shishkarev_a_sum_of_vector_elements_mpi::MPIVectorSumParallel parallel(task_data_par);
  ASSERT_TRUE(parallel.ValidationImpl());
  parallel.PreProcessingImpl();
  parallel.RunImpl();
  parallel.PostProcessingImpl();

  if (world.rank() == 0) {
    ASSERT_EQ(global_sum[0], 42);
  }
}

TEST(shishkarev_a_sum_of_vector_elements_mpi, test_large_vector_sum) {
  boost::mpi::environment env;
  boost::mpi::communicator world;

  std::vector<int> global_vec;
  std::vector<int32_t> global_sum(1, 0);

  std::shared_ptr<ppc::core::TaskData> task_data_par = std::make_shared<ppc::core::TaskData>();
  const int vector_size = 1000000;
  if (world.rank() == 0) {
    global_vec = GetRandomVector(vector_size);
    task_data_par->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vec.data()));
    task_data_par->inputs_count.emplace_back(global_vec.size());
    task_data_par->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_sum.data()));
    task_data_par->outputs_count.emplace_back(global_sum.size());
  }

  shishkarev_a_sum_of_vector_elements_mpi::MPIVectorSumParallel parallel(task_data_par);
  ASSERT_TRUE(parallel.ValidationImpl());
  parallel.PreProcessingImpl();
  parallel.RunImpl();
  parallel.PostProcessingImpl();

  if (world.rank() == 0) {
    int expected_sum = std::accumulate(global_vec.begin(), global_vec.end(), 0);
    ASSERT_EQ(global_sum[0], expected_sum);
  }
}

TEST(shishkarev_a_sum_of_vector_elements_mpi, test_zero_vector_sum) {
  boost::mpi::environment env;
  boost::mpi::communicator world;

  std::vector<int> global_vec(100, 0);
  std::vector<int32_t> global_sum(1, 0);

  std::shared_ptr<ppc::core::TaskData> task_data_par = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    task_data_par->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vec.data()));
    task_data_par->inputs_count.emplace_back(global_vec.size());
    task_data_par->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_sum.data()));
    task_data_par->outputs_count.emplace_back(global_sum.size());
  }

  shishkarev_a_sum_of_vector_elements_mpi::MPIVectorSumParallel parallel(task_data_par);
  ASSERT_TRUE(parallel.ValidationImpl());
  parallel.PreProcessingImpl();
  parallel.RunImpl();
  parallel.PostProcessingImpl();

  if (world.rank() == 0) {
    ASSERT_EQ(global_sum[0], 0);
  }
}

TEST(shishkarev_a_sum_of_vector_elements_mpi, test_negative_vector_sum) {
  boost::mpi::environment env;
  boost::mpi::communicator world;

  std::vector<int> global_vec(100, -1);
  std::vector<int32_t> global_sum(1, 0);

  std::shared_ptr<ppc::core::TaskData> task_data_par = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    task_data_par->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vec.data()));
    task_data_par->inputs_count.emplace_back(global_vec.size());
    task_data_par->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_sum.data()));
    task_data_par->outputs_count.emplace_back(global_sum.size());
  }

  shishkarev_a_sum_of_vector_elements_mpi::MPIVectorSumParallel parallel(task_data_par);
  ASSERT_TRUE(parallel.ValidationImpl());
  parallel.PreProcessingImpl();
  parallel.RunImpl();
  parallel.PostProcessingImpl();

  if (world.rank() == 0) {
    int expected_sum = std::accumulate(global_vec.begin(), global_vec.end(), 0);
    ASSERT_EQ(global_sum[0], expected_sum);
  }
}