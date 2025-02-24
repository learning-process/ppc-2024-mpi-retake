// Copyright 2023 Nesterov Alexander
#include <gtest/gtest.h>

#include <vector>

#include "seq/leontev_n_average/include/ops_seq.hpp"

template <class InOutType>
void taskEmplacement(std::shared_ptr<ppc::core::TaskData> &taskDataPar, std::vector<InOutType> &global_vec,
                     std::vector<InOutType> &global_avg) {
  taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(global_vec.data()));
  taskDataPar->inputs_count.emplace_back(global_vec.size());
  taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(global_avg.data()));
  taskDataPar->outputs_count.emplace_back(global_avg.size());
}

TEST(leontev_n_average_seq, int_vector_avg) {
  // Create data
  std::vector<int32_t> in(5, 10);
  const int32_t expected_avg = 10;
  std::vector<int32_t> out(1, 0);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskEmplacement<int32_t>(taskDataSeq, in, out);

  // Create Task
  leontev_n_average_seq::VecAvgSequential<int32_t> vecAvgSequential(taskDataSeq);
  ASSERT_TRUE(vecAvgSequential.ValidationImpl());
  vecAvgSequential.PreProcessingImpl();
  vecAvgSequential.RunImpl();
  vecAvgSequential.PostProcessingImpl();
  ASSERT_EQ(expected_avg, out[0]);
}

TEST(leontev_n_average_seq, double_vector_avg) {
  // Create data
  std::vector<double> in(5, 10.0);
  const double expected_avg = 10.0;
  std::vector<double> out(1, 0.0);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskEmplacement<double>(taskDataSeq, in, out);

  // Create Task
  leontev_n_average_seq::VecAvgSequential<double> vecAvgSequential(taskDataSeq);
  ASSERT_TRUE(vecAvgSequential.ValidationImpl());
  vecAvgSequential.PreProcessingImpl();
  vecAvgSequential.RunImpl();
  vecAvgSequential.PostProcessingImpl();
  EXPECT_NEAR(out[0], expected_avg, 1e-6);
}

TEST(leontev_n_average_seq, float_vector_avg) {
  // Create data
  std::vector<float> in(5, 1.f);
  std::vector<float> out(1, 0.f);
  const float expected_avg = 1.f;

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskEmplacement<float>(taskDataSeq, in, out);

  // Create Task
  leontev_n_average_seq::VecAvgSequential<float> vecAvgSequential(taskDataSeq);
  ASSERT_TRUE(vecAvgSequential.ValidationImpl());
  vecAvgSequential.PreProcessingImpl();
  vecAvgSequential.RunImpl();
  vecAvgSequential.PostProcessingImpl();
  EXPECT_NEAR(out[0], expected_avg, 1e-3f);
}

TEST(leontev_n_average_seq, int32_vector_avg) {
  // Create data
  std::vector<int32_t> in(2000, 5);
  in[0] = 3;
  in[1] = 7;
  std::vector<int32_t> out(1, 0);
  const int32_t expected_avg = 5;

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskEmplacement<int32_t>(taskDataSeq, in, out);

  // Create Task
  leontev_n_average_seq::VecAvgSequential<int32_t> vecAvgSequential(taskDataSeq);
  ASSERT_TRUE(vecAvgSequential.ValidationImpl());
  vecAvgSequential.PreProcessingImpl();
  vecAvgSequential.RunImpl();
  vecAvgSequential.PostProcessingImpl();
  ASSERT_EQ(out[0], expected_avg);
}

TEST(leontev_n_average_seq, uint32_vector_avg) {
  // Create data
  std::vector<uint32_t> in(255, 2);
  in[0] = 0;
  in[1] = 4;
  std::vector<uint32_t> out(1, 0);
  const uint32_t expected_avg = 2;

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskEmplacement<uint32_t>(taskDataSeq, in, out);

  // Create Task
  leontev_n_average_seq::VecAvgSequential<uint32_t> vecAvgSequential(taskDataSeq);
  ASSERT_TRUE(vecAvgSequential.ValidationImpl());
  vecAvgSequential.PreProcessingImpl();
  vecAvgSequential.RunImpl();
  vecAvgSequential.PostProcessingImpl();
  ASSERT_EQ(out[0], expected_avg);
}

TEST(leontev_n_average_seq, vector_avg_0) {
  // Create data
  std::vector<int32_t> in(1, 0);
  const int32_t expected_avg = 0;
  std::vector<int32_t> out(1, 0);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskEmplacement<int32_t>(taskDataSeq, in, out);

  // Create Task
  leontev_n_average_seq::VecAvgSequential<int32_t> vecAvgSequential(taskDataSeq);
  ASSERT_TRUE(vecAvgSequential.ValidationImpl());
  vecAvgSequential.PreProcessingImpl();
  vecAvgSequential.RunImpl();
  vecAvgSequential.PostProcessingImpl();
  ASSERT_EQ(expected_avg, out[0]);
}
