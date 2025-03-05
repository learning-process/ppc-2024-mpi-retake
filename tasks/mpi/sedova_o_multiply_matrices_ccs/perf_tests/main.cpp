#include <boost/mpi/communicator.hpp>
#include <boost/mpi/environment.hpp>
#include <boost/mpi/timer.hpp>
#include <random>
#include <vector>

#include "boost/mpi/collectives/broadcast.hpp"
#include "core/perf/include/perf.hpp"
#include "core/task/include/task.hpp"
#include "mpi/sedova_o_multiply_matrices_ccs/include/ops_mpi.hpp"

namespace sedova_o_multiply_matrices_ccs_mpi {
namespace {
std::vector<std::vector<double>> GenerateMatrix(int rows, int cols, int non_zero_count) {
  std::vector<std::vector<double>> matrix(rows, std::vector<double>(cols, 0.0));
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<> row_dist(0, rows - 1);
  std::uniform_int_distribution<> col_dist(0, cols - 1);
  std::uniform_real_distribution<> value_dist(-10.0, 10.0);

  int count = 0;
  while (count < non_zero_count) {
    int r = row_dist(gen);
    int c = col_dist(gen);

    if (matrix[r][c] == 0.0) {
      matrix[r][c] = value_dist(gen);
      ++count;
    }
  }

  return matrix;
}

std::vector<std::vector<double>> MultiplyMatrices(const std::vector<std::vector<double>> &a,
                                                  const std::vector<std::vector<double>> &b) {
  int rows_A_ = a.size();
  int cols_A_ = a[0].size();
  int cols_B_ = b[0].size();

  std::vector<std::vector<double>> result(rows_A_, std::vector<double>(cols_B_, 0.0));

  for (int i = 0; i < rows_A_; ++i) {
    for (int j = 0; j < cols_B_; ++j) {
      for (int k = 0; k < cols_A_; ++k) {
        result[i][j] += a[i][k] * b[k][j];
      }
    }
  }

  return result;
}
}  // namespace
}  // namespace sedova_o_multiply_matrices_ccs_mpi

TEST(sedova_o_multiply_matrices_ccs_mpi, test_pipeline_run) {
  boost::mpi::communicator world;
  int size = 256;
  int elements = 6553;
  std::vector<std::vector<double>> a_;
  std::vector<std::vector<double>> b_;

  if (world.rank() == 0) {
    a_ = sedova_o_multiply_matrices_ccs_mpi::GenerateMatrix(size, size, elements);
    b_ = sedova_o_multiply_matrices_ccs_mpi::GenerateMatrix(size, size, elements);
  }
  boost::mpi::broadcast(world, a_, 0);
  boost::mpi::broadcast(world, b_, 0);
  std::vector<double> A_val;
  std::vector<int> A_row_ind;
  std::vector<int> A_col_ptr;
  int rows_A_ = a_.size();
  int cols_A_ = a_[0].size();

  std::vector<double> B_val;
  std::vector<int> B_row_ind;
  std::vector<int> B_col_ptr;
  int rows_B_ = b_.size();
  int cols_B_ = b_[0].size();

  std::vector<double> exp_C_val;
  std::vector<int> exp_C_row_ind;
  std::vector<int> exp_C_col_ptr;

  if (world.rank() == 0) {
    auto exp_C = sedova_o_multiply_matrices_ccs_mpi::MultiplyMatrices(a_, b_);
    sedova_o_multiply_matrices_ccs_mpi::Convertirovanie(exp_C, exp_C.size(), exp_C[0].size(), exp_C_val, exp_C_row_ind,
                                                        exp_C_col_ptr);
  }
  std::vector<double> C_val;
  std::vector<int> C_row_ind;
  std::vector<int> C_col_ptr;

  sedova_o_multiply_matrices_ccs_mpi::Convertirovanie(a_, rows_A_, cols_A_, A_val, A_row_ind, A_col_ptr);
  sedova_o_multiply_matrices_ccs_mpi::Convertirovanie(b_, rows_B_, cols_B_, B_val, B_row_ind, B_col_ptr);

  std::shared_ptr<ppc::core::TaskData> task_data = std::make_shared<ppc::core::TaskData>();

  task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(&rows_A_));
  task_data->inputs_count.emplace_back(1);
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(&cols_A_));
  task_data->inputs_count.emplace_back(1);
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(&rows_B_));
  task_data->inputs_count.emplace_back(1);
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(&cols_B_));
  task_data->inputs_count.emplace_back(1);

  if (world.rank() == 0) {
    task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(A_val.data()));
    task_data->inputs_count.emplace_back(A_val.size());
    task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(A_row_ind.data()));
    task_data->inputs_count.emplace_back(A_row_ind.size());
    task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(A_col_ptr.data()));
    task_data->inputs_count.emplace_back(A_col_ptr.size());

    task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(B_val.data()));
    task_data->inputs_count.emplace_back(B_val.size());
    task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(B_row_ind.data()));
    task_data->inputs_count.emplace_back(B_row_ind.size());
    task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(B_col_ptr.data()));
    task_data->inputs_count.emplace_back(B_col_ptr.size());

    C_val.resize(exp_C_val.size());
    C_row_ind.resize(exp_C_row_ind.size());
    C_col_ptr.resize(exp_C_col_ptr.size());

    task_data->outputs.emplace_back(reinterpret_cast<uint8_t *>(C_val.data()));
    task_data->outputs_count.emplace_back(C_val.size());
    task_data->outputs.emplace_back(reinterpret_cast<uint8_t *>(C_row_ind.data()));
    task_data->outputs_count.emplace_back(C_row_ind.size());
    task_data->outputs.emplace_back(reinterpret_cast<uint8_t *>(C_col_ptr.data()));
    task_data->outputs_count.emplace_back(C_col_ptr.size());
  }

  auto task = std::make_shared<sedova_o_multiply_matrices_ccs_mpi::TestTaskMPI>(task_data);

  auto perf_attr = std::make_shared<ppc::core::PerfAttr>();
  perf_attr->num_running = 10;
  const auto t0 = std::chrono::high_resolution_clock::now();
  perf_attr->current_timer = [&] {
    auto current_time_point = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count();
    return static_cast<double>(duration) * 1e-9;
  };
  auto perf_results = std::make_shared<ppc::core::PerfResults>();
  auto perf_analyzer = std::make_shared<ppc::core::Perf>(task);
  perf_analyzer->TaskRun(perf_attr, perf_results);
  if (world.rank() == 0) {
    ppc::core::Perf::PrintPerfStatistic(perf_results);
    ASSERT_EQ(exp_C_val, C_val);
    ASSERT_EQ(exp_C_row_ind, C_row_ind);
    ASSERT_EQ(exp_C_col_ptr, C_col_ptr);
  }
}