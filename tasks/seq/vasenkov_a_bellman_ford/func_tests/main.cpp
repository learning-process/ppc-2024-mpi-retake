#include <gtest/gtest.h>

#include <cstdint>
#include <limits>
#include <memory>
#include <vector>

#include "core/task/include/task.hpp"
#include "seq/vasenkov_a_bellman_ford/include/ops_seq.hpp"

TEST(vasenkov_a_bellman_ford_seq, simple_graph) {
  std::vector<int> row_ptr = {0, 2, 4, 5, 5};
  std::vector<int> col_ind = {1, 2, 2, 3, 3};
  std::vector<int> weights = {4, 5, -3, 2, 1};
  int num_vertices = 4;
  int source_vertex = 0;

  std::vector<int> expected_distances = {0, 4, 1, 2};

  std::vector<int> output_distances(num_vertices, std::numeric_limits<int>::max());

  std::shared_ptr<ppc::core::TaskData> task_data_seq = std::make_shared<ppc::core::TaskData>();

  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(row_ptr.data()));
  task_data_seq->inputs_count.emplace_back(row_ptr.size());

  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(col_ind.data()));
  task_data_seq->inputs_count.emplace_back(col_ind.size());

  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(weights.data()));
  task_data_seq->inputs_count.emplace_back(weights.size());

  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&num_vertices));
  task_data_seq->inputs_count.emplace_back(1);

  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&source_vertex));
  task_data_seq->inputs_count.emplace_back(1);

  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(output_distances.data()));
  task_data_seq->outputs_count.emplace_back(output_distances.size());

  vasenkov_a_bellman_ford_seq::BellmanFordSequential task_sequential(task_data_seq);
  ASSERT_TRUE(task_sequential.ValidationImpl());
  task_sequential.PreProcessingImpl();
  ASSERT_TRUE(task_sequential.RunImpl());
  task_sequential.PostProcessingImpl();

  ASSERT_EQ(output_distances, expected_distances);
}

TEST(vasenkov_a_bellman_ford_seq, graph_with_negative_cycle) {
  std::vector<int> row_ptr = {0, 2, 3, 4};
  std::vector<int> col_ind = {1, 2, 0, 1};
  std::vector<int> weights = {1, -1, -2, -1};
  int num_vertices = 3;
  int source_vertex = 0;

  std::vector<int> output_distances(num_vertices, std::numeric_limits<int>::max());

  std::shared_ptr<ppc::core::TaskData> task_data_seq = std::make_shared<ppc::core::TaskData>();

  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(row_ptr.data()));
  task_data_seq->inputs_count.emplace_back(row_ptr.size());

  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(col_ind.data()));
  task_data_seq->inputs_count.emplace_back(col_ind.size());

  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(weights.data()));
  task_data_seq->inputs_count.emplace_back(weights.size());

  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&num_vertices));
  task_data_seq->inputs_count.emplace_back(1);

  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&source_vertex));
  task_data_seq->inputs_count.emplace_back(1);

  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(output_distances.data()));
  task_data_seq->outputs_count.emplace_back(output_distances.size());

  vasenkov_a_bellman_ford_seq::BellmanFordSequential task_sequential(task_data_seq);
  ASSERT_TRUE(task_sequential.ValidationImpl());
  task_sequential.PreProcessingImpl();
  EXPECT_FALSE(task_sequential.RunImpl());
  task_sequential.PostProcessingImpl();
}

TEST(vasenkov_a_bellman_ford_seq, disconnected_graph) {
  std::vector<int> row_ptr = {0, 0, 0, 0, 0};
  std::vector<int> col_ind = {};
  std::vector<int> weights = {};
  int num_vertices = 4;
  int source_vertex = 0;

  std::vector<int> expected_distances = {0, std::numeric_limits<int>::max(), std::numeric_limits<int>::max(),
                                         std::numeric_limits<int>::max()};

  std::vector<int> output_distances(num_vertices, std::numeric_limits<int>::max());

  std::shared_ptr<ppc::core::TaskData> task_data_seq = std::make_shared<ppc::core::TaskData>();

  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(row_ptr.data()));
  task_data_seq->inputs_count.emplace_back(row_ptr.size());

  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(col_ind.data()));
  task_data_seq->inputs_count.emplace_back(col_ind.size());

  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(weights.data()));
  task_data_seq->inputs_count.emplace_back(weights.size());

  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&num_vertices));
  task_data_seq->inputs_count.emplace_back(1);

  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&source_vertex));
  task_data_seq->inputs_count.emplace_back(1);

  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(output_distances.data()));
  task_data_seq->outputs_count.emplace_back(output_distances.size());

  vasenkov_a_bellman_ford_seq::BellmanFordSequential task_sequential(task_data_seq);
  ASSERT_TRUE(task_sequential.ValidationImpl());
  task_sequential.PreProcessingImpl();
  ASSERT_TRUE(task_sequential.RunImpl());
  task_sequential.PostProcessingImpl();

  ASSERT_EQ(output_distances, expected_distances);
}
TEST(vasenkov_a_bellman_ford_seq, no_edges) {
  std::vector<int> row_ptr = {0, 0, 0, 0};
  std::vector<int> col_ind = {};
  std::vector<int> weights = {};
  int num_vertices = 3;
  int source_vertex = 0;

  std::vector<int> expected_distances = {0, std::numeric_limits<int>::max(), std::numeric_limits<int>::max()};

  std::vector<int> output_distances(num_vertices, std::numeric_limits<int>::max());

  std::shared_ptr<ppc::core::TaskData> task_data_seq = std::make_shared<ppc::core::TaskData>();

  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(row_ptr.data()));
  task_data_seq->inputs_count.emplace_back(row_ptr.size());

  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(col_ind.data()));
  task_data_seq->inputs_count.emplace_back(col_ind.size());

  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(weights.data()));
  task_data_seq->inputs_count.emplace_back(weights.size());

  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&num_vertices));
  task_data_seq->inputs_count.emplace_back(1);

  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&source_vertex));
  task_data_seq->inputs_count.emplace_back(1);

  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(output_distances.data()));
  task_data_seq->outputs_count.emplace_back(output_distances.size());

  vasenkov_a_bellman_ford_seq::BellmanFordSequential task_sequential(task_data_seq);
  ASSERT_TRUE(task_sequential.ValidationImpl());
  task_sequential.PreProcessingImpl();
  ASSERT_TRUE(task_sequential.RunImpl());
  task_sequential.PostProcessingImpl();

  ASSERT_EQ(output_distances, expected_distances);
}
TEST(vasenkov_a_bellman_ford_seq, single_edge) {
  std::vector<int> row_ptr = {0, 1, 1};
  std::vector<int> col_ind = {1};
  std::vector<int> weights = {5};
  int num_vertices = 2;
  int source_vertex = 0;

  std::vector<int> expected_distances = {0, 5};

  std::vector<int> output_distances(num_vertices, std::numeric_limits<int>::max());

  std::shared_ptr<ppc::core::TaskData> task_data_seq = std::make_shared<ppc::core::TaskData>();

  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(row_ptr.data()));
  task_data_seq->inputs_count.emplace_back(row_ptr.size());

  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(col_ind.data()));
  task_data_seq->inputs_count.emplace_back(col_ind.size());

  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(weights.data()));
  task_data_seq->inputs_count.emplace_back(weights.size());

  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&num_vertices));
  task_data_seq->inputs_count.emplace_back(1);

  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&source_vertex));
  task_data_seq->inputs_count.emplace_back(1);

  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(output_distances.data()));
  task_data_seq->outputs_count.emplace_back(output_distances.size());

  vasenkov_a_bellman_ford_seq::BellmanFordSequential task_sequential(task_data_seq);
  ASSERT_TRUE(task_sequential.ValidationImpl());
  task_sequential.PreProcessingImpl();
  ASSERT_TRUE(task_sequential.RunImpl());
  task_sequential.PostProcessingImpl();

  ASSERT_EQ(output_distances, expected_distances);
}
TEST(vasenkov_a_bellman_ford_seq, negative_weights_no_negative_cycle) {
  std::vector<int> row_ptr = {0, 2, 4, 5, 5};
  std::vector<int> col_ind = {1, 2, 2, 3, 3};
  std::vector<int> weights = {4, 5, -3, 2, 1};
  int num_vertices = 4;
  int source_vertex = 0;

  std::vector<int> expected_distances = {0, 4, -1};

  std::vector<int> output_distances(num_vertices, std::numeric_limits<int>::max());

  std::shared_ptr<ppc::core::TaskData> task_data_seq = std::make_shared<ppc::core::TaskData>();

  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(row_ptr.data()));
  task_data_seq->inputs_count.emplace_back(row_ptr.size());

  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(col_ind.data()));
  task_data_seq->inputs_count.emplace_back(col_ind.size());

  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(weights.data()));
  task_data_seq->inputs_count.emplace_back(weights.size());

  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&num_vertices));
  task_data_seq->inputs_count.emplace_back(1);

  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&source_vertex));
  task_data_seq->inputs_count.emplace_back(1);

  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(output_distances.data()));
  task_data_seq->outputs_count.emplace_back(output_distances.size());

  vasenkov_a_bellman_ford_seq::BellmanFordSequential task_sequential(task_data_seq);
  ASSERT_TRUE(task_sequential.ValidationImpl());
  task_sequential.PreProcessingImpl();
  ASSERT_TRUE(task_sequential.RunImpl());
  task_sequential.PostProcessingImpl();

  ASSERT_EQ(output_distances, expected_distances);
}
TEST(vasenkov_a_bellman_ford_seq, negative_cycle) {
  std::vector<int> row_ptr = {0, 2, 3};
  std::vector<int> col_ind = {1, 0, 0};
  std::vector<int> weights = {1, -2, -1};
  int num_vertices = 2;
  int source_vertex = 0;

  std::vector<int> output_distances(num_vertices, std::numeric_limits<int>::max());

  std::shared_ptr<ppc::core::TaskData> task_data_seq = std::make_shared<ppc::core::TaskData>();

  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(row_ptr.data()));
  task_data_seq->inputs_count.emplace_back(row_ptr.size());

  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(col_ind.data()));
  task_data_seq->inputs_count.emplace_back(col_ind.size());

  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(weights.data()));
  task_data_seq->inputs_count.emplace_back(weights.size());

  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&num_vertices));
  task_data_seq->inputs_count.emplace_back(1);

  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&source_vertex));
  task_data_seq->inputs_count.emplace_back(1);

  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(output_distances.data()));
  task_data_seq->outputs_count.emplace_back(output_distances.size());

  vasenkov_a_bellman_ford_seq::BellmanFordSequential task_sequential(task_data_seq);
  ASSERT_TRUE(task_sequential.ValidationImpl());
  task_sequential.PreProcessingImpl();
  ASSERT_FALSE(task_sequential.RunImpl());
}
