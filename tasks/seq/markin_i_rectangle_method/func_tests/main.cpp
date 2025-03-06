#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <fstream>
#include <memory>
#include <vector>

#include "core/task/include/task.hpp"
#include "core/util/include/util.hpp"
#include "seq/markin_i_rectangle_method/include/ops_seq.hpp"

TEST(markin_i_rectangle_method_seq, test_simple) {


  float left = 1;
  float right = 4;
  int steps = 1000;
  float out = 0;




  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&left));
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&right));
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&steps));
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(&out));


  markin_i_rectangle_method_seq::RectangleSequential test_task_sequential(task_data_seq);
  ASSERT_EQ(test_task_sequential.Validation(), true);
  test_task_sequential.PreProcessing();
  test_task_sequential.Run();
  test_task_sequential.PostProcessing();
  ASSERT_NEAR(15.9375, out, 1e-2);
}
TEST(markin_i_rectangle_method_seq, test_negativ_borders) {

  float left = -7;
  float right = -2;
  int steps = 1000;
  float out = 0;



  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&left));
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&right));
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&steps));
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(&out));


  markin_i_rectangle_method_seq::RectangleSequential test_task_sequential(task_data_seq);
  ASSERT_EQ(test_task_sequential.Validation(), true);
  test_task_sequential.PreProcessing();
  test_task_sequential.Run();
  test_task_sequential.PostProcessing();
  ASSERT_NEAR(-149.0625, out, 1e-2);
}