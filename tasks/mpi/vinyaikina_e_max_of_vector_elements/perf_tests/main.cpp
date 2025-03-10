
#include <gtest/gtest.h>

#include <boost/mpi/communicator.hpp>
#include <boost/mpi/timer.hpp>
#include <boost/serialization/vector.hpp>  //NOLINT
#include <cstdint>
#include <limits>
#include <memory>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "core/task/include/task.hpp"
#include "mpi/vinyaikina_e_max_of_vector_elements/include/ops_mpi.hpp"

TEST(vinyaikina_e_max_of_vector_elements_mpi, test_pipeline_run) {
  boost::mpi::communicator world;
  std::vector<int32_t> input_vector;
  int32_t result_parallel = std::numeric_limits<int32_t>::min();
  auto task_data_par = std::make_shared<ppc::core::TaskData>();
  int vector_size = 50000000;

  if (world.rank() == 0) {
    input_vector.resize(vector_size, 1);
    input_vector[vector_size / 2] = 10;
    task_data_par->inputs.emplace_back(reinterpret_cast<uint8_t*>(input_vector.data()));
    task_data_par->inputs_count.emplace_back(input_vector.size());
    task_data_par->outputs.emplace_back(reinterpret_cast<uint8_t*>(&result_parallel));
    task_data_par->outputs_count.emplace_back(1);
  }

  auto test_mpi_task_parallel = std::make_shared<vinyaikina_e_max_of_vector_elements::VectorMaxPar>(task_data_par);

  auto perf_attr = std::make_shared<ppc::core::PerfAttr>();
  perf_attr->num_running = 10;
  const boost::mpi::timer current_timer;
  perf_attr->current_timer = [&] { return current_timer.elapsed(); };
  auto perf_results = std::make_shared<ppc::core::PerfResults>();
  auto perf_analyzer = std::make_shared<ppc::core::Perf>(test_mpi_task_parallel);

  perf_analyzer->PipelineRun(perf_attr, perf_results);

  if (world.rank() == 0) {
    ppc::core::Perf::PrintPerfStatistic(perf_results);
    ASSERT_EQ(10, result_parallel);
  }
}

TEST(vinyaikina_e_max_of_vector_elements_mpi, test_task_run) {
  boost::mpi::communicator world;
  std::vector<int32_t> input_vector;
  int32_t result_parallel = std::numeric_limits<int32_t>::min();
  auto task_data_par = std::make_shared<ppc::core::TaskData>();
  int vector_size = 50000000;

  if (world.rank() == 0) {
    input_vector.resize(vector_size, 1);
    input_vector[0] = -5;
    task_data_par->inputs.emplace_back(reinterpret_cast<uint8_t*>(input_vector.data()));
    task_data_par->inputs_count.emplace_back(input_vector.size());
    task_data_par->outputs.emplace_back(reinterpret_cast<uint8_t*>(&result_parallel));
    task_data_par->outputs_count.emplace_back(1);
  }

  auto test_mpi_task_parallel = std::make_shared<vinyaikina_e_max_of_vector_elements::VectorMaxPar>(task_data_par);

  auto perf_attr = std::make_shared<ppc::core::PerfAttr>();
  perf_attr->num_running = 10;
  const boost::mpi::timer current_timer;
  perf_attr->current_timer = [&] { return current_timer.elapsed(); };
  auto perf_results = std::make_shared<ppc::core::PerfResults>();
  auto perf_analyzer = std::make_shared<ppc::core::Perf>(test_mpi_task_parallel);

  perf_analyzer->TaskRun(perf_attr, perf_results);

  if (world.rank() == 0) {
    ppc::core::Perf::PrintPerfStatistic(perf_results);
    ASSERT_EQ(1, result_parallel);
  }
}
