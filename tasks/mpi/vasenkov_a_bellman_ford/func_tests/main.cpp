#include <gtest/gtest.h>

#include <boost/mpi/collectives/broadcast.hpp>
#include <boost/mpi/communicator.hpp>
#include <cstdint>
#include <memory>
#include <random>
#include <vector>

#include "core/task/include/task.hpp"
#include "mpi/vasenkov_a_bellman_ford/include/ops_mpi.hpp"

namespace {
void GenerateRandomGraph(int num_vertices, int edges_per_vertex, std::vector<int> &row_ptr, std::vector<int> &col_ind,
                         std::vector<int> &weights, int min_weight = 1, int max_weight = 10) {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<int> weight_dist(min_weight, max_weight);
  std::uniform_int_distribution<int> vertex_dist(0, num_vertices - 1);

  row_ptr.clear();
  col_ind.clear();
  weights.clear();

  row_ptr.push_back(0);
  for (int u = 0; u < num_vertices; ++u) {
    for (int e = 0; e < edges_per_vertex; ++e) {
      int v = vertex_dist(gen);
      if (v != u) {
        col_ind.push_back(v);
        weights.push_back(weight_dist(gen));
      }
    }
    row_ptr.push_back(static_cast<uint8_t>(col_ind.size()));
    row_ptr.push_back(static_cast<int>(col_ind.size()));
  }
}

void RunSequentialVersion(const std::vector<int> &row_ptr, const std::vector<int> &col_ind,
                          const std::vector<int> &weights, int num_vertices, int source_vertex,
                          std::vector<int> &seq_result) {
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();

  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(const_cast<int *>(row_ptr.data())));
  task_data_seq->inputs_count.emplace_back(row_ptr.size());

  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(const_cast<int *>(col_ind.data())));
  task_data_seq->inputs_count.emplace_back(col_ind.size());

  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(const_cast<int *>(weights.data())));
  task_data_seq->inputs_count.emplace_back(weights.size());

  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&num_vertices));
  task_data_seq->inputs_count.emplace_back(1);

  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&source_vertex));
  task_data_seq->inputs_count.emplace_back(1);

  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(seq_result.data()));
  task_data_seq->outputs_count.emplace_back(seq_result.size());

  auto task_sequential = std::make_shared<vasenkov_a_bellman_ford_mpi::BellmanFordSequentialMPI>(task_data_seq);
  task_sequential->ValidationImpl();
  task_sequential->PreProcessingImpl();
  bool seq_run_res = task_sequential->RunImpl();
  task_sequential->PostProcessingImpl();

  ASSERT_TRUE(seq_run_res);
}

bool RunParallelVersion(const std::vector<int> &row_ptr, const std::vector<int> &col_ind,
                        const std::vector<int> &weights, int num_vertices, int source_vertex,
                        std::vector<int> &global_result) {
  boost::mpi::communicator world;

  std::shared_ptr<ppc::core::TaskData> task_data_par = std::make_shared<ppc::core::TaskData>();
  task_data_par->inputs.emplace_back(reinterpret_cast<uint8_t *>(const_cast<int *>(row_ptr.data())));
  task_data_par->inputs_count.emplace_back(row_ptr.size());

  task_data_par->inputs.emplace_back(reinterpret_cast<uint8_t *>(const_cast<int *>(col_ind.data())));
  task_data_par->inputs_count.emplace_back(col_ind.size());

  task_data_par->inputs.emplace_back(reinterpret_cast<uint8_t *>(const_cast<int *>(weights.data())));
  task_data_par->inputs_count.emplace_back(weights.size());

  task_data_par->inputs.emplace_back(reinterpret_cast<uint8_t *>(&num_vertices));
  task_data_par->inputs_count.emplace_back(1);

  task_data_par->inputs.emplace_back(reinterpret_cast<uint8_t *>(&source_vertex));
  task_data_par->inputs_count.emplace_back(1);

  task_data_par->outputs.emplace_back(reinterpret_cast<uint8_t *>(global_result.data()));
  task_data_par->outputs_count.emplace_back(global_result.size());

  auto task_parallel = std::make_shared<vasenkov_a_bellman_ford_mpi::BellmanFordMPI>(task_data_par);
  task_parallel->ValidationImpl();
  task_parallel->PreProcessingImpl();
  bool par_run_res = task_parallel->RunImpl();
  task_parallel->PostProcessingImpl();

  return par_run_res;
}
}  // namespace

TEST(vasenkov_a_bellman_ford_mpi, simple_graph0) {
  boost::mpi::communicator world;

  std::vector<int> row_ptr = {0, 2, 4, 5, 5};
  std::vector<int> col_ind = {1, 2, 2, 3, 3};
  std::vector<int> weights = {4, 5, -3, 2, 1};
  int num_vertices = 4;
  int source_vertex = 0;
  std::vector<int> global_result(num_vertices);

  ASSERT_EQ(row_ptr.size(), num_vertices + 1);
  ASSERT_EQ(col_ind.size(), row_ptr[num_vertices]);
  ASSERT_EQ(weights.size(), row_ptr[num_vertices]);

  RunParallelVersion(row_ptr, col_ind, weights, num_vertices, source_vertex, global_result);

  if (world.rank() == 0) {
    std::vector<int> seq_result(num_vertices);
    RunSequentialVersion(row_ptr, col_ind, weights, num_vertices, source_vertex, seq_result);
    ASSERT_EQ(global_result, seq_result);
  }
}

TEST(vasenkov_a_bellman_ford_mpi, simple_graph1) {
  boost::mpi::communicator world;

  std::vector<int> row_ptr = {0, 1, 1};
  std::vector<int> col_ind = {1};
  std::vector<int> weights = {5};
  int num_vertices = 2;
  int source_vertex = 0;
  std::vector<int> global_result(num_vertices);

  ASSERT_EQ(row_ptr.size(), num_vertices + 1);
  ASSERT_EQ(col_ind.size(), row_ptr[num_vertices]);
  ASSERT_EQ(weights.size(), row_ptr[num_vertices]);

  RunParallelVersion(row_ptr, col_ind, weights, num_vertices, source_vertex, global_result);

  if (world.rank() == 0) {
    std::vector<int> seq_result(num_vertices);
    RunSequentialVersion(row_ptr, col_ind, weights, num_vertices, source_vertex, seq_result);
    ASSERT_EQ(global_result, seq_result);
  }
}

TEST(vasenkov_a_bellman_ford_mpi, simple_graph2) {
  boost::mpi::communicator world;

  std::vector<int> row_ptr = {0, 2, 3, 4};
  std::vector<int> col_ind = {1, 2, 2, 0};
  std::vector<int> weights = {4, -1, -2, 3};
  int num_vertices = 3;
  int source_vertex = 0;
  std::vector<int> global_result(num_vertices);

  ASSERT_EQ(row_ptr.size(), num_vertices + 1);
  ASSERT_EQ(col_ind.size(), row_ptr[num_vertices]);
  ASSERT_EQ(weights.size(), row_ptr[num_vertices]);

  RunParallelVersion(row_ptr, col_ind, weights, num_vertices, source_vertex, global_result);

  if (world.rank() == 0) {
    std::vector<int> seq_result(num_vertices);
    RunSequentialVersion(row_ptr, col_ind, weights, num_vertices, source_vertex, seq_result);
    ASSERT_EQ(global_result, seq_result);
  }
}

TEST(vasenkov_a_bellman_ford_mpi, negative_weights) {
  boost::mpi::communicator world;

  std::vector<int> row_ptr = {0, 2, 4, 6};
  std::vector<int> col_ind = {1, 2, 0, 2, 0, 1};
  std::vector<int> weights = {1, -4, 1, 2, -4, 2};
  int num_vertices = 3;
  int source_vertex = 0;
  std::vector<int> global_result(num_vertices);

  bool has_negative_cycle = !RunParallelVersion(row_ptr, col_ind, weights, num_vertices, source_vertex, global_result);

  if (world.rank() == 0) {
    ASSERT_TRUE(has_negative_cycle);
  }
}

TEST(vasenkov_a_bellman_ford_mpi, random_graph) {
  boost::mpi::communicator world;
  std::vector<int> row_ptr;
  std::vector<int> col_ind;
  std::vector<int> weights;
  int row_ptr_size = 0;
  int col_ind_size = 0;
  int weights_size = 0;

  int num_vertices = 10;
  int source_vertex = 0;
  std::vector<int> global_result(num_vertices);

  if (world.rank() == 0) {
    GenerateRandomGraph(num_vertices, 3, row_ptr, col_ind, weights);
  }

  if (world.rank() == 0) {
    row_ptr_size = static_cast<int>(row_ptr.size());
    col_ind_size = static_cast<int>(col_ind.size());
    weights_size = static_cast<int>(weights.size());
  }

  boost::mpi::broadcast(world, row_ptr_size, 0);
  boost::mpi::broadcast(world, col_ind_size, 0);
  boost::mpi::broadcast(world, weights_size, 0);

  if (world.rank() != 0) {
    row_ptr.resize(row_ptr_size);
    col_ind.resize(col_ind_size);
    weights.resize(weights_size);
  }

  boost::mpi::broadcast(world, row_ptr.data(), row_ptr_size, 0);
  boost::mpi::broadcast(world, col_ind.data(), col_ind_size, 0);
  boost::mpi::broadcast(world, weights.data(), weights_size, 0);

  RunParallelVersion(row_ptr, col_ind, weights, num_vertices, source_vertex, global_result);

  if (world.rank() == 0) {
    std::vector<int> seq_result(num_vertices);
    RunSequentialVersion(row_ptr, col_ind, weights, num_vertices, source_vertex, seq_result);
    ASSERT_EQ(global_result, seq_result);
  }
}