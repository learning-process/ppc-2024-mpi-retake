#include <gtest/gtest.h>

#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <memory>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "core/task/include/task.hpp"
#include "seq/Konstantinov_I_Gauss_Jordan_method/include/ops_seq.hpp"

namespace konstantinov_i_gauss_jordan_method_seq {

  std::vector<double> GetRandomMatrix(int rows, int cols) {
    std::random_device dev;
    std::mt19937 gen(dev());
    std::uniform_real_distribution<double> dist(-20.0, 20.0);
    std::vector<double> matrix(rows * cols);
    for (int i = 0; i < rows * cols; i++) {
      matrix[i] = dist(gen);
    }
    return matrix;
  }

}  // namespace konstantinov_i_gauss_jordan_method_seq

TEST(konstantinov_i_gauss_jordan_method_seq, pipeline_run) {
  int n = 50;
  std::vector<double> global_matrix = konstantinov_i_gauss_jordan_method_seq::GetRandomMatrix(n, n + 1);
  std::vector<double> global_result(n * (n + 1));

  auto taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_matrix.data()));
  taskDataSeq->inputs_count.emplace_back(global_matrix.size());

  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&n));
  taskDataSeq->inputs_count.emplace_back(1);

  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_result.data()));
  taskDataSeq->outputs_count.emplace_back(global_result.size());

  auto taskSequential =
      std::make_shared<konstantinov_i_gauss_jordan_method_seq::GaussJordanMethodSeq>(taskDataSeq);
  EXPECT_TRUE(taskSequential->ValidationImpl());
  taskSequential->PreProcessingImpl();
  taskSequential->RunImpl();
  taskSequential->PostProcessingImpl();

  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;
  auto start_time = std::chrono::high_resolution_clock::now();
  perfAttr->current_timer = [&start_time] {
    auto now = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = now - start_time;
    return elapsed.count();
  };

  auto perfResults = std::make_shared<ppc::core::PerfResults>();

  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(taskSequential);
  perfAnalyzer->PipelineRun(perfAttr, perfResults);

  ppc::core::Perf::PrintPerfStatistic(perfResults);
}

TEST(konstantinov_i_gauss_jordan_method_seq, task_run) {
  int n = 50;
  std::vector<double> global_matrix = konstantinov_i_gauss_jordan_method_seq::GetRandomMatrix(n, n + 1);
  std::vector<double> global_result(n * (n + 1));

  auto taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_matrix.data()));
  taskDataSeq->inputs_count.emplace_back(global_matrix.size());

  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&n));
  taskDataSeq->inputs_count.emplace_back(1);

  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_result.data()));
  taskDataSeq->outputs_count.emplace_back(global_result.size());

  auto taskSequential =
      std::make_shared<konstantinov_i_gauss_jordan_method_seq::GaussJordanMethodSeq>(taskDataSeq);
  EXPECT_TRUE(taskSequential->ValidationImpl());
  taskSequential->PreProcessingImpl();
  taskSequential->RunImpl();
  taskSequential->PostProcessingImpl();

  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;
  auto start_time = std::chrono::high_resolution_clock::now();
  perfAttr->current_timer = [&start_time] {
    auto now = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = now - start_time;
    return elapsed.count();
  };

  auto perfResults = std::make_shared<ppc::core::PerfResults>();

  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(taskSequential);
  perfAnalyzer->TaskRun(perfAttr, perfResults);

  ppc::core::Perf::PrintPerfStatistic(perfResults);
}