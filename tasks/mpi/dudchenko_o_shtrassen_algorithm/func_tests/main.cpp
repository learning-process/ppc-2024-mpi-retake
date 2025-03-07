#include <gtest/gtest.h>

#include <boost/mpi/communicator.hpp>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <random>
#include <vector>

#include "core/task/include/task.hpp"
#include "mpi/dudchenko_o_shtrassen_algorithm/include/ops_mpi.hpp"

namespace {
struct Value {
  double min_value;
  double max_value;
};

struct Parametre {
  std::vector<double> a;
  std::vector<double> b;
};

std::vector<double> GenerateRandomSquareMatrix(size_t n, Value value) {
  std::vector<double> matrix(n * n);

  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<double> dis(value.min_value, value.max_value);

  for (int i = 0; i < static_cast<int>(n * n); ++i) {
    matrix[i] = dis(gen);
  }
  return matrix;
}

void RunSequentialTest(const std::shared_ptr<ppc::core::TaskData> &task_data) {
  dudchenko_o_shtrassen_algorithm_mpi::StrassenAlgoriphmSequential strassen_matrix_mult_seq(task_data);
  ASSERT_TRUE(strassen_matrix_mult_seq.ValidationImpl());
  ASSERT_TRUE(strassen_matrix_mult_seq.PreProcessingImpl());
  ASSERT_TRUE(strassen_matrix_mult_seq.RunImpl());
  ASSERT_TRUE(strassen_matrix_mult_seq.PostProcessingImpl());
}

void RunParallelTest(const std::shared_ptr<ppc::core::TaskData> &task_data) {
  dudchenko_o_shtrassen_algorithm_mpi::StrassenAlgoriphmParallel strassen_matrix_mult_par(task_data);
  ASSERT_TRUE(strassen_matrix_mult_par.ValidationImpl());
  ASSERT_TRUE(strassen_matrix_mult_par.PreProcessingImpl());
  ASSERT_TRUE(strassen_matrix_mult_par.RunImpl());
  ASSERT_TRUE(strassen_matrix_mult_par.PostProcessingImpl());
}

void CreateTest(size_t n) {
  boost::mpi::communicator world;
  std::vector<double> a = GenerateRandomSquareMatrix(n, {.min_value = -200, .max_value = 200});
  std::vector<double> b = GenerateRandomSquareMatrix(n, {.min_value = -200, .max_value = 200});

  std::vector<double> out_seq(n * n, 0.0);
  std::vector<double> out_par(n * n, 0.0);

  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  auto task_data_par = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(a.data()));
    task_data_seq->inputs_count.emplace_back(a.size());
    task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(b.data()));
    task_data_seq->inputs_count.emplace_back(b.size());
    task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out_seq.data()));
    task_data_seq->outputs_count.emplace_back(out_seq.size());
    RunSequentialTest(task_data_seq);
  }

  if (world.rank() == 0) {
    task_data_par->inputs.emplace_back(reinterpret_cast<uint8_t *>(a.data()));
    task_data_par->inputs_count.emplace_back(a.size());
    task_data_par->inputs.emplace_back(reinterpret_cast<uint8_t *>(b.data()));
    task_data_par->inputs_count.emplace_back(b.size());
    task_data_par->outputs.emplace_back(reinterpret_cast<uint8_t *>(out_par.data()));
    task_data_par->outputs_count.emplace_back(out_par.size());
  }

  RunParallelTest(task_data_par);

  for (size_t i = 0; i < n * n; i++) {
    ASSERT_NEAR(out_seq[i], out_par[i], 1e-8);
  }
}
}  // namespace

TEST(dudchenko_o_shtrassen_algorithm_mpi, test_2x2_matrices) { CreateTest(2); }

TEST(dudchenko_o_shtrassen_algorithm_mpi, test_3x3_matrices) { CreateTest(3); }

TEST(dudchenko_o_shtrassen_algorithm_mpi, test_4x4_matrices) { CreateTest(4); }

TEST(dudchenko_o_shtrassen_algorithm_mpi, test_5x5_matrices) { CreateTest(5); }

TEST(dudchenko_o_shtrassen_algorithm_mpi, test_10x10_matrices) { CreateTest(10); }

TEST(dudchenko_o_shtrassen_algorithm_mpi, test_12x12_matrices) { CreateTest(12); }

TEST(dudchenko_o_shtrassen_algorithm_mpi, test_15x15_matrices) { CreateTest(15); }

TEST(dudchenko_o_shtrassen_algorithm_mpi, test_23x23_matrices) { CreateTest(23); }

TEST(dudchenko_o_shtrassen_algorithm_mpi, test_25x25_matrices) { CreateTest(25); }

TEST(dudchenko_o_shtrassen_algorithm_mpi, test_50x50_matrices) { CreateTest(50); }

TEST(dudchenko_o_shtrassen_algorithm_mpi, test_64x64_matrices) { CreateTest(64); }

TEST(dudchenko_o_shtrassen_algorithm_mpi, test_different_size_matrices) {
  boost::mpi::communicator world;
  std::vector<double> a = {1.0, 2.0, 3.0, 4.0};
  std::vector<double> b = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};

  auto task_data_par = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    task_data_par->inputs.emplace_back(reinterpret_cast<uint8_t *>(a.data()));
    task_data_par->inputs_count.emplace_back(a.size());
    task_data_par->inputs.emplace_back(reinterpret_cast<uint8_t *>(b.data()));
    task_data_par->inputs_count.emplace_back(b.size());

    dudchenko_o_shtrassen_algorithm_mpi::StrassenAlgoriphmParallel strassen_matrix_mult_par(task_data_par);
    ASSERT_FALSE(strassen_matrix_mult_par.ValidationImpl());
  }
}

TEST(dudchenko_o_shtrassen_algorithm_mpi, test_non_squared_matrices) {
  boost::mpi::communicator world;
  std::vector<double> a = {1.0, 2.0, 3.0, 4.0, 5.0};
  std::vector<double> b = {6.0, 7.0, 8.0, 9.0, 10.0};

  auto task_data_par = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    task_data_par->inputs.emplace_back(reinterpret_cast<uint8_t *>(a.data()));
    task_data_par->inputs_count.emplace_back(a.size());
    task_data_par->inputs.emplace_back(reinterpret_cast<uint8_t *>(b.data()));
    task_data_par->inputs_count.emplace_back(b.size());

    dudchenko_o_shtrassen_algorithm_mpi::StrassenAlgoriphmParallel strassen_matrix_mult_par(task_data_par);
    ASSERT_FALSE(strassen_matrix_mult_par.ValidationImpl());
  }
}

TEST(dudchenko_o_shtrassen_algorithm_mpi, test_non_valid_input) {
  boost::mpi::communicator world;
  std::vector<double> a = {1.0, 2.0, 3.0, 4.0, 5.0};
  std::vector<double> b = {6.0, 7.0, 8.0, 9.0, 10.0};

  auto task_data_par = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    task_data_par->inputs.emplace_back(reinterpret_cast<uint8_t *>(a.data()));
    task_data_par->inputs_count.emplace_back(a.size());
    task_data_par->inputs_count.emplace_back(b.size());

    dudchenko_o_shtrassen_algorithm_mpi::StrassenAlgoriphmParallel strassen_matrix_mult_par(task_data_par);
    ASSERT_FALSE(strassen_matrix_mult_par.ValidationImpl());
  }
}

TEST(dudchenko_o_shtrassen_algorithm_mpi, test_non_valid_outputs_size) {
  boost::mpi::communicator world;

  const size_t n = 2;
  std::vector<double> a = {1.0, 2.0, 3.0, 4.0};
  std::vector<double> b = {5.0, 6.0, 7.0, 8.0};
  std::vector<double> out_par(n, 0.0);  // we need n * n

  auto task_data_par = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    task_data_par->inputs.emplace_back(reinterpret_cast<uint8_t *>(a.data()));
    task_data_par->inputs_count.emplace_back(a.size());
    task_data_par->inputs.emplace_back(reinterpret_cast<uint8_t *>(b.data()));
    task_data_par->inputs_count.emplace_back(b.size());
    task_data_par->outputs.emplace_back(reinterpret_cast<uint8_t *>(out_par.data()));
    task_data_par->outputs_count.emplace_back(out_par.size());

    dudchenko_o_shtrassen_algorithm_mpi::StrassenAlgoriphmParallel strassen_matrix_mult_par(task_data_par);
    ASSERT_FALSE(strassen_matrix_mult_par.ValidationImpl());
  }
}