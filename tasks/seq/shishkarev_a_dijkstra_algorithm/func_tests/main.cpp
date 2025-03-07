// Copyright 2023 Nesterov Alexander
#include <gtest/gtest.h>

#include <cstdint>
#include <memory>
#include <vector>

#include "core/task/include/task.hpp"
#include "seq/shishkarev_a_dijkstra_algorithm/include/ops_seq.hpp"

TEST(shishkarev_a_dijkstra_algorithm_seq, Test_Graph_3x3) {
  int size = 3;
  int st = 0;
  // Create data
  std::vector<int> matrix = {0, 2, 5, 4, 0, 2, 3, 1, 0};
  std::vector<int> res(size, 0);
  std::vector<int> ans = {0, 2, 4};

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix.data()));
  task_data_seq->inputs_count.emplace_back(matrix.size());
  task_data_seq->inputs_count.emplace_back(size);
  task_data_seq->inputs_count.emplace_back(st);
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(res.data()));
  task_data_seq->outputs_count.emplace_back(res.size());

  // Create Task
  shishkarev_a_dijkstra_algorithm_seq::TestTaskSequential test_task_sequential(task_data_seq);
  ASSERT_EQ(test_task_sequential.ValidationImpl(), true);
  test_task_sequential.PreProcessingImpl();
  test_task_sequential.RunImpl();
  test_task_sequential.PostProcessingImpl();
  ASSERT_EQ(ans, res);
}

TEST(shishkarev_a_dijkstra_algorithm_seq, Test_Graph_4x4) {
  int size = 4;
  int st = 0;
  // Create data
  std::vector<int> matrix = {0, 9, 9, 3, 6, 0, 3, 5, 1, 3, 0, 5, 2, 2, 10, 0};
  std::vector<int> res(size, 0);
  std::vector<int> ans = {0, 5, 8, 3};

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix.data()));
  task_data_seq->inputs_count.emplace_back(matrix.size());
  task_data_seq->inputs_count.emplace_back(size);
  task_data_seq->inputs_count.emplace_back(st);
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(res.data()));
  task_data_seq->outputs_count.emplace_back(res.size());

  // Create Task
  shishkarev_a_dijkstra_algorithm_seq::TestTaskSequential test_task_sequential(task_data_seq);
  ASSERT_EQ(test_task_sequential.ValidationImpl(), true);
  test_task_sequential.PreProcessingImpl();
  test_task_sequential.RunImpl();
  test_task_sequential.PostProcessingImpl();
  ASSERT_EQ(ans, res);
}

TEST(shishkarev_a_dijkstra_algorithm_seq, Test_Graph_5x5) {
  int size = 5;
  int st = 0;
  // Create data
  std::vector<int> matrix = {0, 5, 0, 3, 0, 0, 0, 4, 2, 2, 0, 0, 0, 3, 0, 0, 3, 0, 0, 2, 9, 0, 1, 0, 0};
  std::vector<int> res(size, 0);
  std::vector<int> ans = {0, 5, 6, 3, 5};

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix.data()));
  task_data_seq->inputs_count.emplace_back(matrix.size());
  task_data_seq->inputs_count.emplace_back(size);
  task_data_seq->inputs_count.emplace_back(st);
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(res.data()));
  task_data_seq->outputs_count.emplace_back(res.size());

  // Create Task
  shishkarev_a_dijkstra_algorithm_seq::TestTaskSequential test_task_sequential(task_data_seq);
  ASSERT_EQ(test_task_sequential.ValidationImpl(), true);
  test_task_sequential.PreProcessingImpl();
  test_task_sequential.RunImpl();
  test_task_sequential.PostProcessingImpl();
  ASSERT_EQ(ans, res);
}

TEST(shishkarev_a_dijkstra_algorithm_seq, Test_Negative_Value) {
  int size = 3;
  int st = 0;
  // Create data
  std::vector<int> matrix = {0, 2, 5, -4, 0, 2, 3, 1, 0};
  std::vector<int> res(size, 0);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix.data()));
  task_data_seq->inputs_count.emplace_back(matrix.size());
  task_data_seq->inputs_count.emplace_back(size);
  task_data_seq->inputs_count.emplace_back(st);
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(res.data()));
  task_data_seq->outputs_count.emplace_back(res.size());

  // Create Task
  shishkarev_a_dijkstra_algorithm_seq::TestTaskSequential test_task_sequential(task_data_seq);
  ASSERT_FALSE(test_task_sequential.ValidationImpl());
}

TEST(shishkarev_a_dijkstra_algorithm_seq, Test_Source_Vertex_False) {
  int size = 3;
  int st = 5;
  // Create data
  std::vector<int> matrix = {0, 2, 5, 4, 0, 2, 3, 1, 0};
  std::vector<int> res(size, 0);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix.data()));
  task_data_seq->inputs_count.emplace_back(matrix.size());
  task_data_seq->inputs_count.emplace_back(size);
  task_data_seq->inputs_count.emplace_back(st);
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(res.data()));
  task_data_seq->outputs_count.emplace_back(res.size());

  // Create Task
  shishkarev_a_dijkstra_algorithm_seq::TestTaskSequential test_task_sequential(task_data_seq);
  ASSERT_FALSE(test_task_sequential.ValidationImpl());
}