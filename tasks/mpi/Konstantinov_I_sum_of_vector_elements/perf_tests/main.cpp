#include <gtest/gtest.h>

#include <boost/mpi/timer.hpp>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

#include "boost/mpi/communicator.hpp"
#include "core/perf/include/perf.hpp"
#include "core/task/include/task.hpp"
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

TEST(Konstantinov_I_sum_of_vector_mpi, test_pipeline_run) {
  boost::mpi::communicator world;
  int rows = 10000;
  int columns = 10000;
  int result;
  std::vector<std::vector<int>> input =
      Konstantinov_I_sum_of_vector_elements_mpi::generate_rand_matrix(rows, columns, 1, 1);
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
  auto test = std::make_shared<Konstantinov_I_sum_of_vector_elements_mpi::SumVecElemParallel>(taskDataPar);

  test->ValidationImpl();
  test->PreProcessingImpl();
  test->RunImpl();
  test->PostProcessingImpl();

  // Create Perf attributes
  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;
  const boost::mpi::timer current_timer;
  perfAttr->current_timer = [&] { return current_timer.elapsed(); };

  // Create and init perf results
  auto perfResults = std::make_shared<ppc::core::PerfResults>();

  // Create Perf analyzer
  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(test);
  perfAnalyzer->PipelineRun(perfAttr, perfResults);

  if (world.rank() == 0) {
    ppc::core::Perf::PrintPerfStatistic(perfResults);
    ASSERT_EQ(rows * columns, result);
  }
}

TEST(Konstantinov_I_sum_of_vector_mpi, test_task_run) {
  boost::mpi::communicator world;
  int rows = 10000;
  int columns = 10000;
  int result;
  std::vector<std::vector<int>> input =
      Konstantinov_I_sum_of_vector_elements_mpi::generate_rand_matrix(rows, columns, 1, 1);
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
  auto test = std::make_shared<Konstantinov_I_sum_of_vector_elements_mpi::SumVecElemParallel>(taskDataPar);

  test->ValidationImpl();
  test->PreProcessingImpl();
  test->RunImpl();
  test->PostProcessingImpl();

  // Create Perf attributes
  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;
  const boost::mpi::timer current_timer;
  perfAttr->current_timer = [&] { return current_timer.elapsed(); };

  // Create and init perf results
  auto perfResults = std::make_shared<ppc::core::PerfResults>();

  // Create Perf analyzer
  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(test);
  perfAnalyzer->TaskRun(perfAttr, perfResults);

  if (world.rank() == 0) {
    ppc::core::Perf::PrintPerfStatistic(perfResults);
    ASSERT_EQ(rows * columns, result);
  }
}