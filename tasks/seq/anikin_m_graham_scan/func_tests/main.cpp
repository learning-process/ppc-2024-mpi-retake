// Anikin Maksim 2025
#include <gtest/gtest.h>

#include <vector>

#include "seq/anikin_m_graham_scan/include/ops_seq.hpp"

TEST(anikin_m_graham_scan, case_0) {
  // Create data
  std::vector<anikin_m_graham_scan_seq::pt> in;
  std::vector<anikin_m_graham_scan_seq::pt> out;

  anikin_m_graham_scan_seq::create_test_data(in, 0);

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

  auto *out_ptr = reinterpret_cast<anikin_m_graham_scan_seq::pt*>(task_data_seq->outputs[0]);
  out = std::vector<anikin_m_graham_scan_seq::pt>(out_ptr, out_ptr + task_data_seq->outputs_count[0]);

  EXPECT_EQ(true, anikin_m_graham_scan_seq::test_data(out, 0));
}

TEST(anikin_m_graham_scan, case_1) {
  // Create data
  std::vector<anikin_m_graham_scan_seq::pt> in;
  std::vector<anikin_m_graham_scan_seq::pt> out;

  anikin_m_graham_scan_seq::create_test_data(in, 1);

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

  auto *out_ptr = reinterpret_cast<anikin_m_graham_scan_seq::pt *>(task_data_seq->outputs[0]);
  out = std::vector<anikin_m_graham_scan_seq::pt>(out_ptr, out_ptr + task_data_seq->outputs_count[0]);

  EXPECT_EQ(true, anikin_m_graham_scan_seq::test_data(out, 1));
}

TEST(anikin_m_graham_scan, case_2) {
  // Create data
  std::vector<anikin_m_graham_scan_seq::pt> in;
  std::vector<anikin_m_graham_scan_seq::pt> out;

  anikin_m_graham_scan_seq::create_test_data(in, 2);

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

  auto *out_ptr = reinterpret_cast<anikin_m_graham_scan_seq::pt *>(task_data_seq->outputs[0]);
  out = std::vector<anikin_m_graham_scan_seq::pt>(out_ptr, out_ptr + task_data_seq->outputs_count[0]);

  EXPECT_EQ(true, anikin_m_graham_scan_seq::test_data(out, 2));
}