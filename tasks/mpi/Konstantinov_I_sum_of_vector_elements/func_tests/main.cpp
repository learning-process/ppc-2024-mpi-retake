#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <fstream>
#include <memory>
#include <string>
#include <vector>

#include "core/task/include/task.hpp"
#include "core/util/include/util.hpp"
#include "mpi/Konstantinov_I_sum_of_vector_elements/include/ops_mpi.hpp"

std::vector<int> Konstantinov_I_sum_of_vector_elements_mpi::generate_rand_vector(int size, int lower_bound,
                                                                                 int upper_bound) {
  std::vector<int> result(size);
  for (int i = 0; i < size; i++) {
    result[i] = lower_bound + rand() % (upper_bound - lower_bound + 1);
  }
  return result;
}

std::vector<std::vector<int>> Konstantinov_I_sum_of_vector_elements_mpi::generate_rand_matrix(int rows, int columns,
                                                                                              int lower_bound,
                                                                                              int upper_bound) {
  std::vector<std::vector<int>> result(rows);
  for (int i = 0; i < rows; i++) {
    result[i] = Konstantinov_I_sum_of_vector_elements_mpi::generate_rand_vector(columns, lower_bound, upper_bound);
  }
  return result;
}

TEST(Konstantinov_I_sum_of_vector_elements_parallel, EmptyInput) {
  boost::mpi::communicator world;
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  Konstantinov_I_sum_of_vector_elements_mpi::SumVecElemParallel test(taskDataPar);
  if (world.rank() == 0) {
    ASSERT_FALSE(test.ValidationImpl());
  }
}

TEST(Konstantinov_I_sum_of_vector_elements_parallel, EmptyOutput) {
  boost::mpi::communicator world;
  int rows = 10;
  int columns = 10;
  std::vector<std::vector<int>> input = Konstantinov_I_sum_of_vector_elements_mpi::generate_rand_matrix(rows, columns);
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    taskDataPar->inputs_count.emplace_back(rows);
    taskDataPar->inputs_count.emplace_back(columns);
    for (long unsigned int i = 0; i < input.size(); i++) {
      taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(input[i].data()));
    }
  }
  Konstantinov_I_sum_of_vector_elements_mpi::SumVecElemParallel test(taskDataPar);
  if (world.rank() == 0) {
    ASSERT_FALSE(test.ValidationImpl());
  }
}

TEST(Konstantinov_I_sum_of_vector_elements_parallel, Matrix1x1) {
  boost::mpi::communicator world;

  int rows = 1;
  int columns = 1;
  int result = 0;
  std::vector<std::vector<int>> input = Konstantinov_I_sum_of_vector_elements_mpi::generate_rand_matrix(rows, columns);
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    taskDataPar->inputs_count.emplace_back(rows);
    taskDataPar->inputs_count.emplace_back(columns);
    for (long unsigned int i = 0; i < input.size(); i++) {
      taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(input[i].data()));
    }
    taskDataPar->outputs_count.emplace_back(1);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(&result));
  }

  Konstantinov_I_sum_of_vector_elements_mpi::SumVecElemParallel test(taskDataPar);
  test.ValidationImpl();
  test.PreProcessingImpl();
  test.RunImpl();
  test.PostProcessingImpl();

  if (world.rank() == 0) {
    int respar = result;
    Konstantinov_I_sum_of_vector_elements_mpi::SumVecElemSequential testseq(taskDataPar);
    testseq.ValidationImpl();
    testseq.PreProcessingImpl();
    testseq.RunImpl();
    testseq.PostProcessingImpl();
    ASSERT_EQ(respar, result);
  }
}

TEST(Konstantinov_I_sum_of_vector_elements_parallel, Matrix5x1) {
  boost::mpi::communicator world;

  int rows = 5;
  int columns = 1;
  int result = 0;
  std::vector<std::vector<int>> input = Konstantinov_I_sum_of_vector_elements_mpi::generate_rand_matrix(rows, columns);
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    taskDataPar->inputs_count.emplace_back(rows);
    taskDataPar->inputs_count.emplace_back(columns);
    for (long unsigned int i = 0; i < input.size(); i++) {
      taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(input[i].data()));
    }
    taskDataPar->outputs_count.emplace_back(1);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(&result));
  }

  Konstantinov_I_sum_of_vector_elements_mpi::SumVecElemParallel test(taskDataPar);
  test.ValidationImpl();
  test.PreProcessingImpl();
  test.RunImpl();
  test.PostProcessingImpl();

  if (world.rank() == 0) {
    int respar = result;
    Konstantinov_I_sum_of_vector_elements_mpi::SumVecElemSequential testseq(taskDataPar);
    testseq.ValidationImpl();
    testseq.PreProcessingImpl();
    testseq.RunImpl();
    testseq.PostProcessingImpl();
    ASSERT_EQ(respar, result);
  }
}

TEST(Konstantinov_I_sum_of_vector_elements_parallel, Matrix10x10) {
  boost::mpi::communicator world;

  int rows = 10;
  int columns = 10;
  int result = 0;
  std::vector<std::vector<int>> input = Konstantinov_I_sum_of_vector_elements_mpi::generate_rand_matrix(rows, columns);
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    taskDataPar->inputs_count.emplace_back(rows);
    taskDataPar->inputs_count.emplace_back(columns);
    for (long unsigned int i = 0; i < input.size(); i++) {
      taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(input[i].data()));
    }
    taskDataPar->outputs_count.emplace_back(1);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(&result));
  }

  Konstantinov_I_sum_of_vector_elements_mpi::SumVecElemParallel test(taskDataPar);
  test.ValidationImpl();
  test.PreProcessingImpl();
  test.RunImpl();
  test.PostProcessingImpl();

  if (world.rank() == 0) {
    int respar = result;
    Konstantinov_I_sum_of_vector_elements_mpi::SumVecElemSequential testseq(taskDataPar);
    testseq.ValidationImpl();
    testseq.PreProcessingImpl();
    testseq.RunImpl();
    testseq.PostProcessingImpl();
    ASSERT_EQ(respar, result);
  }
}

TEST(Konstantinov_I_sum_of_vector_elements_parallel, Matrix100x100) {
  boost::mpi::communicator world;

  int rows = 100;
  int columns = 100;
  int result = 0;
  std::vector<std::vector<int>> input = Konstantinov_I_sum_of_vector_elements_mpi::generate_rand_matrix(rows, columns);
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    taskDataPar->inputs_count.emplace_back(rows);
    taskDataPar->inputs_count.emplace_back(columns);
    for (long unsigned int i = 0; i < input.size(); i++) {
      taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(input[i].data()));
    }
    taskDataPar->outputs_count.emplace_back(1);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(&result));
  }

  Konstantinov_I_sum_of_vector_elements_mpi::SumVecElemParallel test(taskDataPar);
  test.ValidationImpl();
  test.PreProcessingImpl();
  test.RunImpl();
  test.PostProcessingImpl();

  if (world.rank() == 0) {
    int respar = result;
    Konstantinov_I_sum_of_vector_elements_mpi::SumVecElemSequential testseq(taskDataPar);
    testseq.ValidationImpl();
    testseq.PreProcessingImpl();
    testseq.RunImpl();
    testseq.PostProcessingImpl();
    ASSERT_EQ(respar, result);
  }
}

TEST(Konstantinov_I_sum_of_vector_elements_parallel, Matrix100x10) {
  boost::mpi::communicator world;

  int rows = 100;
  int columns = 10;
  int result = 0;
  std::vector<std::vector<int>> input = Konstantinov_I_sum_of_vector_elements_mpi::generate_rand_matrix(rows, columns);
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    taskDataPar->inputs_count.emplace_back(rows);
    taskDataPar->inputs_count.emplace_back(columns);
    for (long unsigned int i = 0; i < input.size(); i++) {
      taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(input[i].data()));
    }
    taskDataPar->outputs_count.emplace_back(1);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(&result));
  }

  Konstantinov_I_sum_of_vector_elements_mpi::SumVecElemParallel test(taskDataPar);
  test.ValidationImpl();
  test.PreProcessingImpl();
  test.RunImpl();
  test.PostProcessingImpl();

  if (world.rank() == 0) {
    int respar = result;
    Konstantinov_I_sum_of_vector_elements_mpi::SumVecElemSequential testseq(taskDataPar);
    testseq.ValidationImpl();
    testseq.PreProcessingImpl();
    testseq.RunImpl();
    testseq.PostProcessingImpl();
    ASSERT_EQ(respar, result);
  }
}

TEST(Konstantinov_I_sum_of_vector_elements_parallel, Matrix10x100) {
  boost::mpi::communicator world;

  int rows = 10;
  int columns = 100;
  int result = 0;
  std::vector<std::vector<int>> input = Konstantinov_I_sum_of_vector_elements_mpi::generate_rand_matrix(rows, columns);
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    taskDataPar->inputs_count.emplace_back(rows);
    taskDataPar->inputs_count.emplace_back(columns);
    for (long unsigned int i = 0; i < input.size(); i++) {
      taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(input[i].data()));
    }
    taskDataPar->outputs_count.emplace_back(1);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(&result));
  }

  Konstantinov_I_sum_of_vector_elements_mpi::SumVecElemParallel test(taskDataPar);
  test.ValidationImpl();
  test.PreProcessingImpl();
  test.RunImpl();
  test.PostProcessingImpl();

  if (world.rank() == 0) {
    int respar = result;
    Konstantinov_I_sum_of_vector_elements_mpi::SumVecElemSequential testseq(taskDataPar);
    testseq.ValidationImpl();
    testseq.PreProcessingImpl();
    testseq.RunImpl();
    testseq.PostProcessingImpl();
    ASSERT_EQ(respar, result);
  }
}