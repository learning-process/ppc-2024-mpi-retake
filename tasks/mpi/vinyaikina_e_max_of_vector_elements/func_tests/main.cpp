#include <gtest/gtest.h>

#include <algorithm>
#include <boost/mpi/communicator.hpp>
#include <cstdint>
#include <limits>
#include <memory>

#include "core/task/include/task.hpp"

void static RunParallelAndSequentialTasks(std::vector<int32_t>& input_vector, int32_t expected_max) {
  boost::mpi::communicator world;
  int32_t result_parallel = std::numeric_limits<int32_t>::min();
  int32_t result_sequential = std::numeric_limits<int32_t>::min();

  std::shared_ptr<ppc::core::TaskData> task_data_par = std::make_shared<ppc::core::TaskData>();
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
    std::shared_ptr<ppc::core::TaskData> task_data_seq = std::make_shared<ppc::core::TaskData>();
    task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(input_vector.data()));
    task_data_seq->inputs_count.emplace_back(input_vector.size());
    task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t*>(&result_sequential));
    task_data_seq->outputs_count.emplace_back(1);

    vinyaikina_e_max_of_vector_elements::VectorMaxSeq test_mpi_task_sequential(task_data_seq);
    test_mpi_task_sequential.ValidationImpl();
    test_mpi_task_sequential.PreProcessingImpl();
    test_mpi_task_sequential.RunImpl();
    test_mpi_task_sequential.PostProcessingImpl();

    ASSERT_EQ(result_sequential, result_parallel);
    ASSERT_EQ(result_sequential, expected_max);
  }
}

TEST(vinyaikina_e_max_of_vector_elements, randomVector50000) {
  boost::mpi::communicator world;
  std::vector<int32_t> input_vector;

  if (world.rank() == 0) {
    input_vector = vinyaikina_e_max_of_vector_elements::MakeRandomVector(50000, -500, 5000);
  }

  boost::mpi::broadcast(world, input_vector, 0);

  int32_t expected_max = 0;
  expected_max = std::ranges::numeric_limits<int32_t>::min();
  if (world.rank() == 0) {
    expected_max = *std::max_element(input_vector.begin(), input_vector.end());
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
  std::shared_ptr<ppc::core::TaskData> task_data = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    task_data->inputs_count.emplace_back(input.size());
    task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(input.data()));
  }

  vinyaikina_e_max_of_vector_elements::VectorMaxPar vector_max_par(task_data);

  if (world.rank() == 0) {
    ASSERT_FALSE(vector_max_par.ValidationImpl());
  }
}