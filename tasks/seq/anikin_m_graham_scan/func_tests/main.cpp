// Anikin Maksim 2025
#include <gtest/gtest.h>

#include <cstdint>
#include <memory>
#include <vector>

#include "core/task/include/task.hpp"
#include "seq/anikin_m_graham_scan/include/ops_seq.hpp"

namespace {
bool TestData(std::vector<anikin_m_graham_scan_seq::Pt> alg_out, int test) {
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

void CreateTestData(std::vector<anikin_m_graham_scan_seq::Pt> &alg_in, int test) {
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

TEST(anikin_m_graham_scan_seq, case_0) {
  // Create data
  std::vector<anikin_m_graham_scan_seq::Pt> in;
  std::vector<anikin_m_graham_scan_seq::Pt> out;

  CreateTestData(in, 0);

  // Create task_data
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_seq->inputs_count.emplace_back(in.size());

  // Create Task
  anikin_m_graham_scan_seq::TestTaskSequential test_task_sequential(task_data_seq);
  ASSERT_EQ(test_task_sequential.Validation(), true);
  test_task_sequential.PreProcessing();
  test_task_sequential.Run();
  test_task_sequential.PostProcessing();

  auto *out_ptr = reinterpret_cast<anikin_m_graham_scan_seq::Pt *>(task_data_seq->outputs[0]);
  out = std::vector<anikin_m_graham_scan_seq::Pt>(out_ptr, out_ptr + task_data_seq->outputs_count[0]);

  EXPECT_EQ(true, TestData(out, 0));
}

TEST(anikin_m_graham_scan_seq, case_1) {
  // Create data
  std::vector<anikin_m_graham_scan_seq::Pt> in;
  std::vector<anikin_m_graham_scan_seq::Pt> out;

  CreateTestData(in, 1);

  // Create task_data
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_seq->inputs_count.emplace_back(in.size());

  // Create Task
  anikin_m_graham_scan_seq::TestTaskSequential test_task_sequential(task_data_seq);
  ASSERT_EQ(test_task_sequential.Validation(), true);
  test_task_sequential.PreProcessing();
  test_task_sequential.Run();
  test_task_sequential.PostProcessing();

  auto *out_ptr = reinterpret_cast<anikin_m_graham_scan_seq::Pt *>(task_data_seq->outputs[0]);
  out = std::vector<anikin_m_graham_scan_seq::Pt>(out_ptr, out_ptr + task_data_seq->outputs_count[0]);

  EXPECT_EQ(true, TestData(out, 1));
}

TEST(anikin_m_graham_scan_seq, case_2) {
  // Create data
  std::vector<anikin_m_graham_scan_seq::Pt> in;
  std::vector<anikin_m_graham_scan_seq::Pt> out;

  CreateTestData(in, 2);

  // Create task_data
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_seq->inputs_count.emplace_back(in.size());

  // Create Task
  anikin_m_graham_scan_seq::TestTaskSequential test_task_sequential(task_data_seq);
  ASSERT_EQ(test_task_sequential.Validation(), true);
  test_task_sequential.PreProcessing();
  test_task_sequential.Run();
  test_task_sequential.PostProcessing();

  auto *out_ptr = reinterpret_cast<anikin_m_graham_scan_seq::Pt *>(task_data_seq->outputs[0]);
  out = std::vector<anikin_m_graham_scan_seq::Pt>(out_ptr, out_ptr + task_data_seq->outputs_count[0]);

  EXPECT_EQ(true, TestData(out, 2));
}