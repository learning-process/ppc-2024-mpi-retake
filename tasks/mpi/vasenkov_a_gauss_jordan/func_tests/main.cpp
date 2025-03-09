#include <gtest/gtest.h>

#include <boost/mpi/communicator.hpp>
#include <cstdint>
#include <memory>
#include <random>
#include <vector>

#include "core/task/include/task.hpp"
#include "mpi/vasenkov_a_gauss_jordan/include/ops_mpi.hpp"

namespace vasenkov_a_gauss_jordan_mpi{
  static std::vector<double> GenerateRandomMatrix(int n, double min_value = -10.0, double max_value = 10.0) {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<double> dist(min_value, max_value);

  std::vector<double> matrix(n * (n + 1));

  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < n + 1; ++j) {
      matrix[(i * (n + 1)) + j] = dist(gen);
    }
  }

  return matrix;
}
std::shared_ptr<ppc::core::TaskData> CreateTaskData(const std::vector<double>& matrix, int n, std::vector<double>& result) {
  auto task_data = std::make_shared<ppc::core::TaskData>();

  if (!matrix.empty()) {
    task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(const_cast<double*>(matrix.data())));
    task_data->inputs_count.emplace_back(matrix.size());
  }

  task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(&n));
  task_data->inputs_count.emplace_back(1);

  if (!result.empty()) {
    task_data->outputs.emplace_back(reinterpret_cast<uint8_t *>(result.data()));
    task_data->outputs_count.emplace_back(result.size());
  }

  return task_data;
}

static void RunSequentialVersion(const std::vector<double> &global_matrix, int n, std::vector<double> &seq_result) {
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();

  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(const_cast<double *>(global_matrix.data())));
  task_data_seq->inputs_count.emplace_back(global_matrix.size());

  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&n));
  task_data_seq->inputs_count.emplace_back(1);

  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(seq_result.data()));
  task_data_seq->outputs_count.emplace_back(seq_result.size());

  auto task_sequential = std::make_shared<vasenkov_a_gauss_jordan_mpi::GaussJordanMethodSequentialMPI>(task_data_seq);
  task_sequential->ValidationImpl();
  task_sequential->PreProcessingImpl();
  bool seq_run_res = task_sequential->RunImpl();
  task_sequential->PostProcessingImpl();

  ASSERT_TRUE(seq_run_res);
}

bool RunParallelVersion(const std::vector<double> &global_matrix, int n, std::vector<double> &global_result) {
  boost::mpi::communicator world;

  std::shared_ptr<ppc::core::TaskData> task_data_par = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    task_data_par->inputs.emplace_back(reinterpret_cast<uint8_t *>(const_cast<double *>(global_matrix.data())));
    task_data_par->inputs_count.emplace_back(global_matrix.size());

    task_data_par->inputs.emplace_back(reinterpret_cast<uint8_t *>(&n));
    task_data_par->inputs_count.emplace_back(1);

    task_data_par->outputs.emplace_back(reinterpret_cast<uint8_t *>(global_result.data()));
    task_data_par->outputs_count.emplace_back(global_result.size());
  }

  auto task_parallel = std::make_shared<vasenkov_a_gauss_jordan_mpi::GaussJordanMethodParallelMPI>(task_data_par);
  task_parallel->ValidationImpl();
  task_parallel->PreProcessingImpl();
  bool par_run_res = task_parallel->RunImpl();
  task_parallel->PostProcessingImpl();

  return par_run_res;
}
}// namespace vasenkov_a_gauss_jordan_mpi

TEST(vasenkov_a_gauss_jordan_mpi, three_simple_matrix) {
  boost::mpi::communicator world;

  std::vector<double> global_matrix;
  int n = 0;
  std::vector<double> global_result;

  if (world.rank() == 0) {
    global_matrix = {1, 2, 1, 10, 4, 8, 3, 20, 2, 5, 9, 30};
    n = 3;

    global_result.resize(n * (n + 1));
  }

  bool par_run_res = vasenkov_a_gauss_jordan_mpi::RunParallelVersion(global_matrix, n, global_result);

  if (world.rank() == 0) {
    std::vector<double> seq_result(global_result.size(), 0);

    vasenkov_a_gauss_jordan_mpi::RunSequentialVersion(global_matrix, n, seq_result);

    if (par_run_res) {
      ASSERT_EQ(global_result.size(), seq_result.size());
      EXPECT_EQ(global_result, seq_result);
    } else {
      EXPECT_FALSE(par_run_res);
    }
  }
}

TEST(vasenkov_a_gauss_jordan_mpi, identity_matrix) {
  boost::mpi::communicator world;

  std::vector<double> global_matrix;
  int n = 0;
  std::vector<double> global_result;

  std::shared_ptr<ppc::core::TaskData> task_data_par = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    global_matrix = {1, 0, 0, 1, 0, 1, 0, 2, 0, 0, 1, 3};
    n = 3;

    global_result.resize(n * (n + 1));
  }

  bool par_run_res = vasenkov_a_gauss_jordan_mpi::RunParallelVersion(global_matrix, n, global_result);

  if (world.rank() == 0) {
    std::vector<double> seq_result(global_result.size(), 0);

    vasenkov_a_gauss_jordan_mpi::RunSequentialVersion(global_matrix, n, seq_result);

    if (par_run_res) {
      ASSERT_EQ(global_result.size(), seq_result.size());
      EXPECT_EQ(global_result, seq_result);
    } else {
      EXPECT_FALSE(par_run_res);
    }
  }
}
TEST(vasenkov_a_gauss_jordan_mpi, singular_matrix) {
  boost::mpi::communicator world;

  std::vector<double> global_matrix;
  int n = 0;
  std::vector<double> global_result;

  std::shared_ptr<ppc::core::TaskData> task_data_par = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    global_matrix = {1, 2, 3, 4, 2, 4, 6, 8, 3, 6, 9, 12};
    n = 3;

    global_result.resize(n * (n + 1));

    task_data_par->inputs.emplace_back(reinterpret_cast<uint8_t *>(const_cast<double *>(global_matrix.data())));
    task_data_par->inputs_count.emplace_back(global_matrix.size());

    task_data_par->inputs.emplace_back(reinterpret_cast<uint8_t *>(&n));
    task_data_par->inputs_count.emplace_back(1);

    task_data_par->outputs.emplace_back(reinterpret_cast<uint8_t *>(global_result.data()));
    task_data_par->outputs_count.emplace_back(global_result.size());
  }

  auto task_parallel = std::make_shared<vasenkov_a_gauss_jordan_mpi::GaussJordanMethodParallelMPI>(task_data_par);
  if (world.rank() == 0) {
    ASSERT_FALSE(task_parallel->ValidationImpl());
  } else {
    ASSERT_TRUE(task_parallel->ValidationImpl());
  }
}
TEST(vasenkov_a_gauss_jordan_mpi, large_matrix) {
  boost::mpi::communicator world;

  std::vector<double> global_matrix;
  int n = 0;
  std::vector<double> global_result;

  std::shared_ptr<ppc::core::TaskData> task_data_par = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    global_matrix = {2, 1, -1, 2, 1, -1, 1, 3, 3, -1, 2, 1, 1, 2, -1, 1, 8, 11, -5, 6};
    n = 4;

    global_result.resize(n * (n + 1));
  }

  bool par_run_res = vasenkov_a_gauss_jordan_mpi::RunParallelVersion(global_matrix, n, global_result);

  if (world.rank() == 0) {
    std::vector<double> seq_result(global_result.size(), 0);

    vasenkov_a_gauss_jordan_mpi:: RunSequentialVersion(global_matrix, n, seq_result);

    if (par_run_res) {
      ASSERT_EQ(global_result.size(), seq_result.size());
      EXPECT_EQ(global_result, seq_result);
    } else {
      EXPECT_FALSE(par_run_res);
    }
  }
}
TEST(vasenkov_a_gauss_jordan_mpi, zero_matrix) {
  boost::mpi::communicator world;

  std::vector<double> global_matrix;
  int n = 0;
  std::vector<double> global_result;

  std::shared_ptr<ppc::core::TaskData> task_data_par = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    global_matrix = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    n = 3;

    global_result.resize(n * (n + 1));

    task_data_par->inputs.emplace_back(reinterpret_cast<uint8_t *>(const_cast<double *>(global_matrix.data())));
    task_data_par->inputs_count.emplace_back(global_matrix.size());

    task_data_par->inputs.emplace_back(reinterpret_cast<uint8_t *>(&n));
    task_data_par->inputs_count.emplace_back(1);

    task_data_par->outputs.emplace_back(reinterpret_cast<uint8_t *>(global_result.data()));
    task_data_par->outputs_count.emplace_back(global_result.size());
  }

  auto task_parallel = std::make_shared<vasenkov_a_gauss_jordan_mpi::GaussJordanMethodParallelMPI>(task_data_par);
  if (world.rank() == 0) {
    ASSERT_FALSE(task_parallel->ValidationImpl());
  } else {
    ASSERT_TRUE(task_parallel->ValidationImpl());
  }
}

TEST(vasenkov_a_gauss_jordan_mpi, random_matrix) {
  boost::mpi::communicator world;

  std::vector<double> global_matrix;
  int n = 0;
  std::vector<double> global_result;

  std::shared_ptr<ppc::core::TaskData> task_data_par = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    n = 4;
    global_matrix = vasenkov_a_gauss_jordan_mpi::GenerateRandomMatrix(n);
    global_result.resize(n * (n + 1));
  }

  bool par_run_res = vasenkov_a_gauss_jordan_mpi::RunParallelVersion(global_matrix, n, global_result);

  if (world.rank() == 0) {
    std::vector<double> seq_result(global_result.size(), 0);

    vasenkov_a_gauss_jordan_mpi::RunSequentialVersion(global_matrix, n, seq_result);

    if (par_run_res) {
      ASSERT_EQ(global_result.size(), seq_result.size());
      EXPECT_EQ(global_result, seq_result);
    } else {
      EXPECT_FALSE(par_run_res);
    }
  }
}