// Copyright 2023 Nesterov Alexander
#include <gtest/gtest.h>

#include <boost/mpi/communicator.hpp>
#include <cstdint>
#include <memory>
#include <random>
#include <vector>

#include "core/task/include/task.hpp"
#include "mpi/shishkarev_a_dijkstra_algorithm/include/ops_mpi.hpp"

namespace {
struct Value {
  int n;
  int min;
  int max;
};

void GenerateMatrix(std::vector<int> &w, Value value) {
  std::random_device dev;
  std::mt19937 gen(dev());
  std::uniform_int_distribution<int> dist(value.min, value.max);
  for (int i = 0; i < value.n * value.n; i++) {
    int val = dist(gen);
    w[i] = val;
  }
  for (int i = 0; i < value.n; i++) {
    w[(i * value.n) + i] = 0;
  }
}
}  // namespace

TEST(shishkarev_a_dijkstra_algorithm_mpi, Test_Graph_5_vertex) {
  boost::mpi::communicator world;
  int size = 5;
  int st = 0;
  int min = 1;
  int max = 10;
  std::vector<int> matrix(size * size, 0);
  std::vector<int> res(size, 0);

  std::shared_ptr<ppc::core::TaskData> task_data_par = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    GenerateMatrix(matrix, {.n = size, .min = min, .max = max});
    task_data_par->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix.data()));
    task_data_par->inputs_count.emplace_back(matrix.size());
    task_data_par->inputs_count.emplace_back(size);
    task_data_par->inputs_count.emplace_back(st);
    task_data_par->outputs.emplace_back(reinterpret_cast<uint8_t *>(res.data()));
    task_data_par->outputs_count.emplace_back(res.size());
  }

  shishkarev_a_dijkstra_algorithm_mpi::TestMPITaskParallel test_mpi_task_parallel(task_data_par);
  ASSERT_EQ(test_mpi_task_parallel.ValidationImpl(), true);
  test_mpi_task_parallel.PreProcessingImpl();
  test_mpi_task_parallel.RunImpl();
  test_mpi_task_parallel.PostProcessingImpl();

  if (world.rank() == 0) {
    std::vector<int> res_seq(size, 0);

    std::shared_ptr<ppc::core::TaskData> task_data_seq = std::make_shared<ppc::core::TaskData>();
    task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix.data()));
    task_data_seq->inputs_count.emplace_back(matrix.size());
    task_data_seq->inputs_count.emplace_back(size);
    task_data_seq->inputs_count.emplace_back(st);
    task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(res_seq.data()));
    task_data_seq->outputs_count.emplace_back(res_seq.size());

    shishkarev_a_dijkstra_algorithm_mpi::TestMPITaskSequential test_mpi_task_sequential(task_data_seq);
    ASSERT_EQ(test_mpi_task_sequential.ValidationImpl(), true);
    test_mpi_task_sequential.PreProcessingImpl();
    test_mpi_task_sequential.RunImpl();
    test_mpi_task_sequential.PostProcessingImpl();

    ASSERT_EQ(res_seq, res);
  }
}

TEST(shishkarev_a_dijkstra_algorithm_mpi, Test_Graph_10_vertex) {
  boost::mpi::communicator world;
  int size = 10;
  int st = 3;
  int min = 5;
  int max = 50;
  std::vector<int> matrix(size * size, 0);
  std::vector<int> res(size, 0);

  std::shared_ptr<ppc::core::TaskData> task_data_par = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    GenerateMatrix(matrix, {.n = size, .min = min, .max = max});
    task_data_par->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix.data()));
    task_data_par->inputs_count.emplace_back(matrix.size());
    task_data_par->inputs_count.emplace_back(size);
    task_data_par->inputs_count.emplace_back(st);
    task_data_par->outputs.emplace_back(reinterpret_cast<uint8_t *>(res.data()));
    task_data_par->outputs_count.emplace_back(res.size());
  }

  shishkarev_a_dijkstra_algorithm_mpi::TestMPITaskParallel test_mpi_task_parallel(task_data_par);
  ASSERT_EQ(test_mpi_task_parallel.ValidationImpl(), true);
  test_mpi_task_parallel.PreProcessingImpl();
  test_mpi_task_parallel.RunImpl();
  test_mpi_task_parallel.PostProcessingImpl();

  if (world.rank() == 0) {
    std::vector<int> res_seq(size, 0);

    std::shared_ptr<ppc::core::TaskData> task_data_seq = std::make_shared<ppc::core::TaskData>();
    task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix.data()));
    task_data_seq->inputs_count.emplace_back(matrix.size());
    task_data_seq->inputs_count.emplace_back(size);
    task_data_seq->inputs_count.emplace_back(st);
    task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(res_seq.data()));
    task_data_seq->outputs_count.emplace_back(res_seq.size());

    shishkarev_a_dijkstra_algorithm_mpi::TestMPITaskSequential test_mpi_task_sequential(task_data_seq);
    ASSERT_EQ(test_mpi_task_sequential.ValidationImpl(), true);
    test_mpi_task_sequential.PreProcessingImpl();
    test_mpi_task_sequential.RunImpl();
    test_mpi_task_sequential.PostProcessingImpl();

    ASSERT_EQ(res_seq, res);
  }
}

TEST(shishkarev_a_dijkstra_algorithm_mpi, Test_Graph_13_vertex) {
  boost::mpi::communicator world;
  int size = 13;
  int st = 3;
  int min = 4;
  int max = 20;
  std::vector<int> matrix(size * size, 0);
  std::vector<int> res(size, 0);

  std::shared_ptr<ppc::core::TaskData> task_data_par = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    GenerateMatrix(matrix, {.n = size, .min = min, .max = max});
    task_data_par->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix.data()));
    task_data_par->inputs_count.emplace_back(matrix.size());
    task_data_par->inputs_count.emplace_back(size);
    task_data_par->inputs_count.emplace_back(st);
    task_data_par->outputs.emplace_back(reinterpret_cast<uint8_t *>(res.data()));
    task_data_par->outputs_count.emplace_back(res.size());
  }

  shishkarev_a_dijkstra_algorithm_mpi::TestMPITaskParallel test_mpi_task_parallel(task_data_par);
  ASSERT_EQ(test_mpi_task_parallel.ValidationImpl(), true);
  test_mpi_task_parallel.PreProcessingImpl();
  test_mpi_task_parallel.RunImpl();
  test_mpi_task_parallel.PostProcessingImpl();

  if (world.rank() == 0) {
    std::vector<int> res_seq(size, 0);

    std::shared_ptr<ppc::core::TaskData> task_data_seq = std::make_shared<ppc::core::TaskData>();
    task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix.data()));
    task_data_seq->inputs_count.emplace_back(matrix.size());
    task_data_seq->inputs_count.emplace_back(size);
    task_data_seq->inputs_count.emplace_back(st);
    task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(res_seq.data()));
    task_data_seq->outputs_count.emplace_back(res_seq.size());

    shishkarev_a_dijkstra_algorithm_mpi::TestMPITaskSequential test_mpi_task_sequential(task_data_seq);
    ASSERT_EQ(test_mpi_task_sequential.ValidationImpl(), true);
    test_mpi_task_sequential.PreProcessingImpl();
    test_mpi_task_sequential.RunImpl();
    test_mpi_task_sequential.PostProcessingImpl();

    ASSERT_EQ(res_seq, res);
  }
}

TEST(shishkarev_a_dijkstra_algorithm_mpi, Test_Graph_20_vertex) {
  boost::mpi::communicator world;
  int size = 20;
  int st = 3;
  int min = 2;
  int max = 40;
  std::vector<int> matrix(size * size, 0);
  std::vector<int> res(size, 0);

  std::shared_ptr<ppc::core::TaskData> task_data_par = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    GenerateMatrix(matrix, {.n = size, .min = min, .max = max});
    task_data_par->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix.data()));
    task_data_par->inputs_count.emplace_back(matrix.size());
    task_data_par->inputs_count.emplace_back(size);
    task_data_par->inputs_count.emplace_back(st);
    task_data_par->outputs.emplace_back(reinterpret_cast<uint8_t *>(res.data()));
    task_data_par->outputs_count.emplace_back(res.size());
  }

  shishkarev_a_dijkstra_algorithm_mpi::TestMPITaskParallel test_mpi_task_parallel(task_data_par);
  ASSERT_EQ(test_mpi_task_parallel.ValidationImpl(), true);
  test_mpi_task_parallel.PreProcessingImpl();
  test_mpi_task_parallel.RunImpl();
  test_mpi_task_parallel.PostProcessingImpl();

  if (world.rank() == 0) {
    std::vector<int> res_seq(size, 0);

    std::shared_ptr<ppc::core::TaskData> task_data_seq = std::make_shared<ppc::core::TaskData>();
    task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix.data()));
    task_data_seq->inputs_count.emplace_back(matrix.size());
    task_data_seq->inputs_count.emplace_back(size);
    task_data_seq->inputs_count.emplace_back(st);
    task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(res_seq.data()));
    task_data_seq->outputs_count.emplace_back(res_seq.size());

    shishkarev_a_dijkstra_algorithm_mpi::TestMPITaskSequential test_mpi_task_sequential(task_data_seq);
    ASSERT_EQ(test_mpi_task_sequential.ValidationImpl(), true);
    test_mpi_task_sequential.PreProcessingImpl();
    test_mpi_task_sequential.RunImpl();
    test_mpi_task_sequential.PostProcessingImpl();

    ASSERT_EQ(res_seq, res);
  }
}

TEST(shishkarev_a_dijkstra_algorithm_mpi, Test_Source_Vertex_False) {
  boost::mpi::communicator world;
  int size = 10;
  int st = 13;
  int min = 2;
  int max = 20;
  std::vector<int> matrix(size * size, 0);
  std::vector<int> res(size, 0);

  std::shared_ptr<ppc::core::TaskData> task_data_par = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    GenerateMatrix(matrix, {.n = size, .min = min, .max = max});
    task_data_par->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix.data()));
    task_data_par->inputs_count.emplace_back(matrix.size());
    task_data_par->inputs_count.emplace_back(size);
    task_data_par->inputs_count.emplace_back(st);
    task_data_par->outputs.emplace_back(reinterpret_cast<uint8_t *>(res.data()));
    task_data_par->outputs_count.emplace_back(res.size());
  }

  shishkarev_a_dijkstra_algorithm_mpi::TestMPITaskParallel test_mpi_task_parallel(task_data_par);
  if (world.rank() == 0) {
    ASSERT_FALSE(test_mpi_task_parallel.ValidationImpl());
  }
}

TEST(shishkarev_a_dijkstra_algorithm_mpi, Test_Negative_Value) {
  boost::mpi::communicator world;
  int size = 3;
  int st = 0;
  std::vector<int> matrix = {0, 2, 5, 4, 0, 2, 3, -1, 0};
  std::vector<int> res(size, 0);

  std::shared_ptr<ppc::core::TaskData> task_data_par = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    task_data_par->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix.data()));
    task_data_par->inputs_count.emplace_back(matrix.size());
    task_data_par->inputs_count.emplace_back(size);
    task_data_par->inputs_count.emplace_back(st);
    task_data_par->outputs.emplace_back(reinterpret_cast<uint8_t *>(res.data()));
    task_data_par->outputs_count.emplace_back(res.size());
  }

  shishkarev_a_dijkstra_algorithm_mpi::TestMPITaskParallel test_mpi_task_parallel(task_data_par);
  if (world.rank() == 0) {
    ASSERT_FALSE(test_mpi_task_parallel.ValidationImpl());
  }
}

TEST(shishkarev_a_dijkstra_algorithm_mpi, Test_Spare_Graph_5x5) {
  boost::mpi::communicator world;
  int size = 5;
  int st = 0;

  std::vector<int> matrix = {0, 5, 0, 3, 0, 0, 0, 4, 2, 2, 0, 0, 0, 3, 0, 0, 3, 0, 0, 2, 9, 0, 1, 0, 0};
  std::vector<int> res(size, 0);

  std::shared_ptr<ppc::core::TaskData> task_data_par = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    task_data_par->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix.data()));
    task_data_par->inputs_count.emplace_back(matrix.size());
    task_data_par->inputs_count.emplace_back(size);
    task_data_par->inputs_count.emplace_back(st);
    task_data_par->outputs.emplace_back(reinterpret_cast<uint8_t *>(res.data()));
    task_data_par->outputs_count.emplace_back(res.size());
  }

  shishkarev_a_dijkstra_algorithm_mpi::TestMPITaskParallel test_mpi_task_parallel(task_data_par);
  ASSERT_EQ(test_mpi_task_parallel.ValidationImpl(), true);
  test_mpi_task_parallel.PreProcessingImpl();
  test_mpi_task_parallel.RunImpl();
  test_mpi_task_parallel.PostProcessingImpl();

  if (world.rank() == 0) {
    std::vector<int> res_seq(size, 0);

    std::shared_ptr<ppc::core::TaskData> task_data_seq = std::make_shared<ppc::core::TaskData>();
    task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix.data()));
    task_data_seq->inputs_count.emplace_back(matrix.size());
    task_data_seq->inputs_count.emplace_back(size);
    task_data_seq->inputs_count.emplace_back(st);
    task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(res_seq.data()));
    task_data_seq->outputs_count.emplace_back(res_seq.size());

    shishkarev_a_dijkstra_algorithm_mpi::TestMPITaskSequential test_mpi_task_sequential(task_data_seq);
    ASSERT_EQ(test_mpi_task_sequential.ValidationImpl(), true);
    test_mpi_task_sequential.PreProcessingImpl();
    test_mpi_task_sequential.RunImpl();
    test_mpi_task_sequential.PostProcessingImpl();

    ASSERT_EQ(res_seq, res);
  }
}