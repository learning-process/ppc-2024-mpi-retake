#include "mpi/somov_i_ribbon_hor_scheme_only_mat_a/include/somov_i_ribbon_hor_scheme_only_mat_a_mpi.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <boost/mpi/communicator.hpp>
#include <chrono>
#include <cstdint>
#include <memory>
#include <random>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "core/task/include/task.hpp"
namespace {
void GetRndVector(std::vector<int> &vec) {
  std::random_device rd;
  std::default_random_engine reng(rd());
  std::uniform_int_distribution<int> dist(-static_cast<int>(vec.size()) - 1, static_cast<int>(vec.size()) - 1);
  std::ranges::generate(vec, [&dist, &reng] { return dist(reng); });
}
}  // namespace
TEST(somov_i_ribbon_hor_scheme_only_mat_a_mpi, test_pipeline_run) {
  boost::mpi::communicator world;
  const int a_c = 500;
  const int a_r = 500;
  const int b_c = 500;
  const int b_r = 500;
  std::vector<int> a;
  std::vector<int> b;
  std::vector<int> c;
  std::shared_ptr<ppc::core::TaskData> task_data = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    a.resize(a_c * a_r);
    b.resize(b_r * b_c);
    c.resize(a_r * b_c);
    GetRndVector(a);
    GetRndVector(b);
    task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(a.data()));
    task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(b.data()));
    task_data->inputs_count.emplace_back(a_c);
    task_data->inputs_count.emplace_back(a_r);
    task_data->inputs_count.emplace_back(b_c);
    task_data->inputs_count.emplace_back(b_r);
    task_data->outputs.emplace_back(reinterpret_cast<uint8_t *>(c.data()));
    task_data->outputs_count.emplace_back(c.size());
  }
  auto test_task_mpi = std::make_shared<somov_i_ribbon_hor_scheme_only_mat_a_mpi::RibbonHorSchemeOnlyMatA>(task_data);
  // Create Perf attributes
  auto perf_attr = std::make_shared<ppc::core::PerfAttr>();
  perf_attr->num_running = 10;
  const auto t0 = std::chrono::high_resolution_clock::now();
  perf_attr->current_timer = [&] {
    auto current_time_point = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count();
    return static_cast<double>(duration) * 1e-9;
  };

  // Create and init perf results
  auto perf_results = std::make_shared<ppc::core::PerfResults>();

  // Create Perf analyzer
  auto perf_analyzer = std::make_shared<ppc::core::Perf>(test_task_mpi);
  perf_analyzer->PipelineRun(perf_attr, perf_results);
  ppc::core::Perf::PrintPerfStatistic(perf_results);
  if (world.rank() == 0) {
    std::vector<int> checker(a_r * b_c, 0);
    somov_i_ribbon_hor_scheme_only_mat_a_mpi::LiterallyMult(a, b, checker, a_c, a_r, b_c);
    ASSERT_EQ(checker, c);
  }
}

TEST(somov_i_ribbon_hor_scheme_only_mat_a_mpi, test_task_run) {
  boost::mpi::communicator world;
  const int a_c = 500;
  const int a_r = 500;
  const int b_c = 500;
  const int b_r = 500;
  std::vector<int> a;
  std::vector<int> b;
  std::vector<int> c;
  std::shared_ptr<ppc::core::TaskData> task_data = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    a.resize(a_c * a_r);
    b.resize(b_r * b_c);
    c.resize(a_r * b_c);
    GetRndVector(a);
    GetRndVector(b);
    task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(a.data()));
    task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(b.data()));
    task_data->inputs_count.emplace_back(a_c);
    task_data->inputs_count.emplace_back(a_r);
    task_data->inputs_count.emplace_back(b_c);
    task_data->inputs_count.emplace_back(b_r);
    task_data->outputs.emplace_back(reinterpret_cast<uint8_t *>(c.data()));
    task_data->outputs_count.emplace_back(c.size());
  }
  auto test_task_mpi = std::make_shared<somov_i_ribbon_hor_scheme_only_mat_a_mpi::RibbonHorSchemeOnlyMatA>(task_data);
  // Create Perf attributes
  auto perf_attr = std::make_shared<ppc::core::PerfAttr>();
  perf_attr->num_running = 10;
  const auto t0 = std::chrono::high_resolution_clock::now();
  perf_attr->current_timer = [&] {
    auto current_time_point = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count();
    return static_cast<double>(duration) * 1e-9;
  };

  // Create and init perf results
  auto perf_results = std::make_shared<ppc::core::PerfResults>();

  // Create Perf analyzer
  auto perf_analyzer = std::make_shared<ppc::core::Perf>(test_task_mpi);
  perf_analyzer->TaskRun(perf_attr, perf_results);
  ppc::core::Perf::PrintPerfStatistic(perf_results);
  if (world.rank() == 0) {
    std::vector<int> checker(a_r * b_c, 0);
    somov_i_ribbon_hor_scheme_only_mat_a_mpi::LiterallyMult(a, b, checker, a_c, a_r, b_c);
    ASSERT_EQ(checker, c);
  }
}
