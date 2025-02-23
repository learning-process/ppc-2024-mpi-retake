#include <cstdint>
#include <gtest/gtest.h>
#include <memory>
#include <vector>

#include "core/task/include/task.hpp"
#include "mpi/muradov_k_trapezoid_integral/include/ops_mpi.hpp"

TEST(muradov_k_trap_integral_mpi, Compare_With_Seq_Result) {
  std::vector<double> input{1.0, 4.0};
  int n = 1e6;
  double seq_result = 21.0;  // (4³ - 1³)/3 = 21
  double mpi_result = 0.0;

  auto task_data_mpi = std::make_shared<ppc::core::TaskData>();
  task_data_mpi->inputs = {reinterpret_cast<uint8_t*>(input.data()),
                           reinterpret_cast<uint8_t*>(&n)};
  task_data_mpi->outputs = {reinterpret_cast<uint8_t*>(&mpi_result)};

  muradov_k_trap_integral_mpi::TrapezoidalIntegral task_mpi(task_data_mpi);
  task_mpi.Run();

  ASSERT_NEAR(seq_result, mpi_result, 1e-3);
}