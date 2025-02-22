#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <fstream>
#include <memory>
#include <string>
#include <vector>

#include "core/task/include/task.hpp"
#include "core/util/include/util.hpp"
#include "mpi/anikin_m_counting_characters/include/ops_mpi.hpp"

TEST(anikin_m_counting_characters_seq, one_char_dif) {
  // Create data
  std::vector<char> in1;
  anikin_m_counting_characters_mpi::create_data_vector(&in1, "aboba");
  std::vector<char> in2;
  anikin_m_counting_characters_mpi::create_data_vector(&in2, "ababa");
  int res_out = 0;

  // Create task_data
  auto task_data_mpi = std::make_shared<ppc::core::TaskData>();
  task_data_mpi->inputs.emplace_back(reinterpret_cast<uint8_t *>(in1.data()));
  task_data_mpi->inputs_count.emplace_back(in1.size());
  task_data_mpi->inputs.emplace_back(reinterpret_cast<uint8_t *>(in2.data()));
  task_data_mpi->inputs_count.emplace_back(in2.size());
  task_data_mpi->outputs.emplace_back(reinterpret_cast<uint8_t *>(&res_out));
  task_data_mpi->outputs_count.emplace_back(1);
  // Create Task
  anikin_m_counting_characters_mpi::TestTaskMPI test_task_mpi(task_data_mpi);
  ASSERT_EQ(test_task_mpi.Validation(), true);
  test_task_mpi.PreProcessing();
  test_task_mpi.Run();
  test_task_mpi.PostProcessing();
  EXPECT_EQ(res_out, 1);
}

TEST(anikin_m_counting_characters_seq, first_larger) {
  // Create data
  std::vector<char> in1;
  anikin_m_counting_characters_mpi::create_data_vector(&in1, "abobaa");
  std::vector<char> in2;
  anikin_m_counting_characters_mpi::create_data_vector(&in2, "ababa");
  int res_out = 0;

  // Create task_data
  auto task_data_mpi = std::make_shared<ppc::core::TaskData>();
  task_data_mpi->inputs.emplace_back(reinterpret_cast<uint8_t *>(in1.data()));
  task_data_mpi->inputs_count.emplace_back(in1.size());
  task_data_mpi->inputs.emplace_back(reinterpret_cast<uint8_t *>(in2.data()));
  task_data_mpi->inputs_count.emplace_back(in2.size());
  task_data_mpi->outputs.emplace_back(reinterpret_cast<uint8_t *>(&res_out));
  task_data_mpi->outputs_count.emplace_back(1);
  // Create Task
  anikin_m_counting_characters_mpi::TestTaskMPI test_task_mpi(task_data_mpi);
  ASSERT_EQ(test_task_mpi.Validation(), true);
  test_task_mpi.PreProcessing();
  test_task_mpi.Run();
  test_task_mpi.PostProcessing();
  EXPECT_EQ(res_out, 2);
}

TEST(anikin_m_counting_characters_seq, second_larger) {
  // Create data
  std::vector<char> in1;
  anikin_m_counting_characters_mpi::create_data_vector(&in1, "aboba");
  std::vector<char> in2;
  anikin_m_counting_characters_mpi::create_data_vector(&in2, "ababaa");
  int res_out = 0;

  // Create task_data
  auto task_data_mpi = std::make_shared<ppc::core::TaskData>();
  task_data_mpi->inputs.emplace_back(reinterpret_cast<uint8_t *>(in1.data()));
  task_data_mpi->inputs_count.emplace_back(in1.size());
  task_data_mpi->inputs.emplace_back(reinterpret_cast<uint8_t *>(in2.data()));
  task_data_mpi->inputs_count.emplace_back(in2.size());
  task_data_mpi->outputs.emplace_back(reinterpret_cast<uint8_t *>(&res_out));
  task_data_mpi->outputs_count.emplace_back(1);
  // Create Task
  anikin_m_counting_characters_mpi::TestTaskMPI test_task_mpi(task_data_mpi);
  ASSERT_EQ(test_task_mpi.Validation(), true);
  test_task_mpi.PreProcessing();
  test_task_mpi.Run();
  test_task_mpi.PostProcessing();
  EXPECT_EQ(res_out, 2);
}