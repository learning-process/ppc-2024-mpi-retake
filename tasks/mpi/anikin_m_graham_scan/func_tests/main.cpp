// Anikin Maksim 2025
#include <gtest/gtest.h>

#include <boost/mpi/communicator.hpp>
#include <cstdint>
#include <memory>
#include <vector>

#include "core/task/include/task.hpp"
#include "mpi/anikin_m_graham_scan/include/ops_mpi.hpp"

namespace {
bool TestData(std::vector<anikin_m_graham_scan_mpi::Pt> alg_out, int test) {
  // case 0
  //  all_points  = [(0, 0), (4, 0), (4, 4), (0, 4), (2, 2)]
  //  hull_points = [(0, 0), (4, 0), (4, 4), (0, 4)]
  // case 1
  //  all_points  = [(0, 0), (1, 1), (2, 2), (3, 3), (4, 0), (4, 4), (0, 4)]
  //  hull_points = [(0, 0), (4, 0), (4, 4), (0, 4)]
  // case 2
  //  all_points  = [(0, 0), (1, 3), (2, 1), (3, 2), (4, 0), (2, 4)]
  //  hull_points = [(0, 0), (4, 0), (2, 4), (1, 3)]
  bool out = true;
  switch (test) {
    case 1:
    case 0:
      if (alg_out.size() == 4) {
        out &= (alg_out[0].x == 0);
        out &= (alg_out[0].y == 0);

        out &= (alg_out[1].x == 0);
        out &= (alg_out[1].y == 4);

        out &= (alg_out[2].x == 4);
        out &= (alg_out[2].y == 4);

        out &= (alg_out[3].x == 4);
        out &= (alg_out[3].y == 0);
      } else {
        out = false;
      }
      break;
    case 2:
      if (alg_out.size() == 4) {
        out &= (alg_out[0].x == 0);
        out &= (alg_out[0].y == 0);

        out &= (alg_out[1].x == 1);
        out &= (alg_out[1].y == 3);

        out &= (alg_out[2].x == 2);
        out &= (alg_out[2].y == 4);

        out &= (alg_out[3].x == 4);
        out &= (alg_out[3].y == 0);
      } else {
        out = false;
      }
      break;
    default:
      break;
  }
  return out;
}

void CreateTestData(std::vector<anikin_m_graham_scan_mpi::Pt> &alg_in, int test) {
  // case 0
  //  all_points  = [(0, 0), (4, 0), (4, 4), (0, 4), (2, 2)]
  //  hull_points = [(0, 0), (4, 0), (4, 4), (0, 4)]
  // case 1
  //  all_points  = [(0, 0), (1, 1), (2, 2), (3, 3), (4, 0), (4, 4), (0, 4)]
  //  hull_points = [(0, 0), (4, 0), (4, 4), (0, 4)]
  // case 2
  //  all_points  = [(0, 0), (1, 3), (2, 1), (3, 2), (4, 0), (2, 4)]
  //  hull_points = [(0, 0), (4, 0), (2, 4), (1, 3)]
  alg_in.clear();
  switch (test) {
    case 0:
      alg_in.push_back({0, 0});
      alg_in.push_back({4, 0});
      alg_in.push_back({4, 4});
      alg_in.push_back({0, 4});
      alg_in.push_back({2, 2});
      break;
    case 1:
      alg_in.push_back({0, 0});
      alg_in.push_back({1, 1});
      alg_in.push_back({2, 2});
      alg_in.push_back({3, 3});
      alg_in.push_back({4, 0});
      alg_in.push_back({4, 4});
      alg_in.push_back({0, 4});
      break;
    case 2:
      alg_in.push_back({0, 0});
      alg_in.push_back({1, 3});
      alg_in.push_back({2, 1});
      alg_in.push_back({3, 2});
      alg_in.push_back({4, 0});
      alg_in.push_back({2, 4});
      break;
    default:
      break;
  }
}
}  // namespace

TEST(anikin_m_graham_scan_mpi, case_0) {
  // Create data
  std::vector<anikin_m_graham_scan_mpi::Pt> in;
  std::vector<anikin_m_graham_scan_mpi::Pt> out;

  CreateTestData(in, 0);

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

  boost::mpi::communicator world;
  if (world.rank() == 0) {
    auto *out_ptr = reinterpret_cast<anikin_m_graham_scan_mpi::Pt *>(task_data_mpi->outputs[0]);
    out = std::vector<anikin_m_graham_scan_mpi::Pt>(out_ptr, out_ptr + task_data_mpi->outputs_count[0]);

    EXPECT_EQ(true, TestData(out, 0));
  } else {
    EXPECT_EQ(true, true);
  }
}

TEST(anikin_m_graham_scan_mpi, case_1) {
  // Create data
  std::vector<anikin_m_graham_scan_mpi::Pt> in;
  std::vector<anikin_m_graham_scan_mpi::Pt> out;

  CreateTestData(in, 1);

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

  boost::mpi::communicator world;
  if (world.rank() == 0) {
    auto *out_ptr = reinterpret_cast<anikin_m_graham_scan_mpi::Pt *>(task_data_mpi->outputs[0]);
    out = std::vector<anikin_m_graham_scan_mpi::Pt>(out_ptr, out_ptr + task_data_mpi->outputs_count[0]);

    EXPECT_EQ(true, TestData(out, 1));
  } else {
    EXPECT_EQ(true, true);
  }
}

TEST(anikin_m_graham_scan_mpi, case_2) {
  // Create data
  std::vector<anikin_m_graham_scan_mpi::Pt> in;
  std::vector<anikin_m_graham_scan_mpi::Pt> out;

  CreateTestData(in, 2);

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

  boost::mpi::communicator world;
  if (world.rank() == 0) {
    auto *out_ptr = reinterpret_cast<anikin_m_graham_scan_mpi::Pt *>(task_data_mpi->outputs[0]);
    out = std::vector<anikin_m_graham_scan_mpi::Pt>(out_ptr, out_ptr + task_data_mpi->outputs_count[0]);

    EXPECT_EQ(true, TestData(out, 2));
  } else {
    EXPECT_EQ(true, true);
  }
}