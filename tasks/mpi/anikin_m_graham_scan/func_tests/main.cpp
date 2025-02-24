// Anikin Maksim 2025
#include <gtest/gtest.h>

#include "core/task/include/task.hpp"
#include "core/util/include/util.hpp"
#include "mpi/anikin_m_graham_scan/include/ops_mpi.hpp"

TEST(anikin_m_graham_scan, case_0) {
  // Create data
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  std::vector<anikin_m_graham_scan_mpi::pt> in;
  std::vector<anikin_m_graham_scan_mpi::pt> out;

  anikin_m_graham_scan_mpi::create_test_data(in, 0);

  // Create task_data
  auto task_data_mpi = std::make_shared<ppc::core::TaskData>();
  task_data_mpi->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_mpi->inputs_count.emplace_back(in.size());

  // Create Task
  anikin_m_graham_scan_mpi::TestTaskMPI test_task_mpi(task_data_mpi);
  ASSERT_EQ(test_task_mpi.Validation(), true);
  test_task_mpi.PreProcessing();
  test_task_mpi.Run();
  test_task_mpi.PostProcessing();

  if (rank == 0) {
    auto *out_ptr = reinterpret_cast<anikin_m_graham_scan_mpi::pt *>(task_data_mpi->outputs[0]);
    out = std::vector<anikin_m_graham_scan_mpi::pt>(out_ptr, out_ptr + task_data_mpi->outputs_count[0]);

    EXPECT_EQ(true, anikin_m_graham_scan_mpi::test_data(out, 0));
  } else {
    EXPECT_EQ(true, true);
  }
}

TEST(anikin_m_graham_scan, case_1) {
  // Create data
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  std::vector<anikin_m_graham_scan_mpi::pt> in;
  std::vector<anikin_m_graham_scan_mpi::pt> out;

  anikin_m_graham_scan_mpi::create_test_data(in, 1);

  // Create task_data
  auto task_data_mpi = std::make_shared<ppc::core::TaskData>();
  task_data_mpi->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_mpi->inputs_count.emplace_back(in.size());

  // Create Task
  anikin_m_graham_scan_mpi::TestTaskMPI test_task_mpi(task_data_mpi);
  ASSERT_EQ(test_task_mpi.Validation(), true);
  test_task_mpi.PreProcessing();
  test_task_mpi.Run();
  test_task_mpi.PostProcessing();

  if (rank == 0) {
    auto *out_ptr = reinterpret_cast<anikin_m_graham_scan_mpi::pt *>(task_data_mpi->outputs[0]);
    out = std::vector<anikin_m_graham_scan_mpi::pt>(out_ptr, out_ptr + task_data_mpi->outputs_count[0]);

    EXPECT_EQ(true, anikin_m_graham_scan_mpi::test_data(out, 1));
  } else {
    EXPECT_EQ(true, true);
  }
}

TEST(anikin_m_graham_scan, case_2) {
  // Create data
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  std::vector<anikin_m_graham_scan_mpi::pt> in;
  std::vector<anikin_m_graham_scan_mpi::pt> out;

  anikin_m_graham_scan_mpi::create_test_data(in, 2);

  // Create task_data
  auto task_data_mpi = std::make_shared<ppc::core::TaskData>();
  task_data_mpi->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_mpi->inputs_count.emplace_back(in.size());

  // Create Task
  anikin_m_graham_scan_mpi::TestTaskMPI test_task_mpi(task_data_mpi);
  ASSERT_EQ(test_task_mpi.Validation(), true);
  test_task_mpi.PreProcessing();
  test_task_mpi.Run();
  test_task_mpi.PostProcessing();

  if (rank == 0) {
    auto *out_ptr = reinterpret_cast<anikin_m_graham_scan_mpi::pt *>(task_data_mpi->outputs[0]);
    out = std::vector<anikin_m_graham_scan_mpi::pt>(out_ptr, out_ptr + task_data_mpi->outputs_count[0]);

    EXPECT_EQ(true, anikin_m_graham_scan_mpi::test_data(out, 2));
  } else {
    EXPECT_EQ(true, true);
  }
}