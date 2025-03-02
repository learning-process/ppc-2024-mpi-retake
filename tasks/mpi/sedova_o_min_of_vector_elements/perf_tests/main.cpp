#include <gtest/gtest.h>

#include <boost/mpi/timer.hpp>
#include <random>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

#include "boost/mpi/communicator.hpp"
#include "core/perf/include/perf.hpp"
#include "core/task/include/task.hpp"
#include "mpi/sedova_o_min_of_vector_elements/include/ops_mpi.hpp"

TEST(sedova_o_min_of_vector_elements_mpi, test_pipeline_run) {
  boost::mpi::communicator world_;
  std::vector<std::vector<int>> global_matrix;
  std::vector<int32_t> global_min(1, INT_MAX);
  int ref = INT_MIN;
  
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> task_data_par = std::make_shared<ppc::core::TaskData>();
  if (world_.rank() == 0) {
    std::random_device dev;
    std::mt19937 gen(dev());
    int rows = 4000;
    int columns = 4000;
    int min = -500;
    int max = 500;

    global_matrix = sedova_o_min_of_vector_elements_mpi::getRandomMatrix(rows, columns, min, max);
    int index = gen() % (rows * columns);
    global_matrix[index / columns][index / rows] = ref;

    for (unsigned int i = 0; i < global_matrix.size(); i++)
      task_data_par->inputs.emplace_back(reinterpret_cast<uint8_t *>(global_matrix[i].data()));
    task_data_par->inputs_count.emplace_back(rows);
    task_data_par->inputs_count.emplace_back(columns);

    task_data_par->outputs.emplace_back(reinterpret_cast<uint8_t *>(global_min.data()));
    task_data_par->outputs_count.emplace_back(global_min.size());
  }

  // Create Task
  sedova_o_min_of_vector_elements_mpi::TestTaskMPI test_task_parallel(task_data_par);
  ASSERT_EQ(test_task_parallel.ValidationImpl(), true);
  test_task_parallel.PreProcessingImpl();
  test_task_parallel.RunImpl();
  test_task_parallel.PostProcessingImpl();

  // Create Perf attributes
  auto perf_attr = std::make_shared<ppc::core::PerfAttr>();
  perf_attr->num_running = 10;
  const boost::mpi::timer current_timer;
  perf_attr->current_timer = [&] { return current_timer.elapsed(); };
  
  // Create and init perf results
  auto perf_results = std::make_shared<ppc::core::PerfResults>();

  // Create Perf analyzer
  auto perf_analyzer = std::make_shared<ppc::core::Perf>(test_task_parallel);
  perf_analyzer->PipelineRun(perf_attr, perf_results);
  if (world_.rank() == 0) {
  ppc::core::Perf::PrintPerfStatistic(perf_results);
  ASSERT_EQ(ref, global_min[0]);
  }
}

TEST(sedova_o_min_of_vector_elements_mpi, test_task_run) {
  boost::mpi::communicator world_;
  std::vector<std::vector<int>> global_matrix;
  std::vector<int32_t> global_min(1, INT_MAX);
  int ref = INT_MIN;

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> task_data_par = std::make_shared<ppc::core::TaskData>();
  if (world_.rank() == 0) {
    std::random_device dev;
    std::mt19937 gen(dev());
    int rows = 4000;
    int columns = 4000;
    int min = -500;
    int max = 500;

    global_matrix = sedova_o_min_of_vector_elements_mpi::getRandomMatrix(rows, columns, min, max);
    int index = gen() % (rows * columns);
    global_matrix[index / columns][index / rows] = ref;

    for (unsigned int i = 0; i < global_matrix.size(); i++)
      task_data_par->inputs.emplace_back(reinterpret_cast<uint8_t *>(global_matrix[i].data()));
    task_data_par->inputs_count.emplace_back(rows);
    task_data_par->inputs_count.emplace_back(columns);

    task_data_par->outputs.emplace_back(reinterpret_cast<uint8_t *>(global_min.data()));
    task_data_par->outputs_count.emplace_back(global_min.size());
  }

  // Create Task
  sedova_o_min_of_vector_elements_mpi::TestTaskMPI test_task_parallel(task_data_par);
  ASSERT_EQ(test_task_parallel.ValidationImpl(), true);
  test_task_parallel.PreProcessingImpl();
  test_task_parallel.RunImpl();
  test_task_parallel.PostProcessingImpl();

  // Create Perf attributes
  auto perf_attr = std::make_shared<ppc::core::PerfAttr>();
  perf_attr->num_running = 10;
  const boost::mpi::timer current_timer;
  perf_attr->current_timer = [&] { return current_timer.elapsed(); };

  // Create and init perf results
  auto perf_results = std::make_shared<ppc::core::PerfResults>();

  // Create Perf analyzer
  auto perf_analyzer = std::make_shared<ppc::core::Perf>(test_task_parallel);
  perf_analyzer->TaskRun(perf_attr, perf_results);
  if (world_.rank() == 0) {
    ppc::core::Perf::PrintPerfStatistic(perf_results);
    ASSERT_EQ(ref, global_min[0]);
  }
}
