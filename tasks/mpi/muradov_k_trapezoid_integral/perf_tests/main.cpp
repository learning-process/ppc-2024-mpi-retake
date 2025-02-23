#include <chrono>
#include <cstdint>
#include <gtest/gtest.h>
#include <memory>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "core/task/include/task.hpp"
#include "mpi/muradov_k_trapezoid_integral/include/ops_mpi.hpp"

TEST(muradov_k_trap_integral_mpi, Perf_Test_Scale) {
  std::vector<double> input{0.0, 10.0};
  int n = 1e8;
  double result = 0.0;

  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs = {reinterpret_cast<uint8_t*>(input.data()),
                      reinterpret_cast<uint8_t*>(&n)};
  task_data->outputs = {reinterpret_cast<uint8_t*>(&result)};

  auto task = std::make_shared<muradov_k_trap_integral_mpi::TrapezoidalIntegral>(task_data);

  auto perf_attr = std::make_shared<ppc::core::PerfAttr>();
  perf_attr->num_running = 5;
  perf_attr->current_timer = [&] {
    return static_cast<double>(std::chrono::high_resolution_clock::now().time_since_epoch().count());
  };

  auto perf_results = std::make_shared<ppc::core::PerfResults>();
  auto perf_analyzer = std::make_shared<ppc::core::Perf>(task);
  perf_analyzer->TaskRun(perf_attr, perf_results);
  ppc::core::Perf::PrintPerfStatistic(perf_results);
}