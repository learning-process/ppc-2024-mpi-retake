#include <gtest/gtest.h>

#include <algorithm>
#include <climits>
#include <cstdint>
#include <memory>
#include <random>
#include <vector>

#include "core/task/include/task.hpp"
#include "seq/sedova_o_min_of_vector_elements/include/ops_seq.hpp"

namespace sedova_o_min_of_vector_elements_seq {
namespace {
std::vector<int> GetRandomVector(int size, int min, int max) {
  std::random_device dev;
  std::mt19937 gen(dev());
  std::uniform_int_distribution<> distrib(min, max);
  std::vector<int> vec(size);
  std::ranges::generate(vec.begin(), vec.end(), [&]() { return distrib(gen); });
  return vec;
}

std::vector<std::vector<int>> GetRandomMatrix(int rows, int columns, int min, int max) {
  std::vector<std::vector<int>> vec(rows);
  std::ranges::generate(vec.begin(), vec.end(), [&]() { return GetRandomVector(columns, min, max); });
  return vec;
}
}  // namespace
}  // namespace sedova_o_min_of_vector_elements_seq

TEST(sedova_o_min_of_vector_elements_seq, test_10x10) {
  std::random_device dev;
  std::mt19937 gen(dev());

  const int rows = 10;
  const int columns = 10;
  const int min = -500;
  const int max = 500;
  int ref = INT_MIN;

  // Create data
  std::vector<int> output(1, INT_MAX);
  std::vector<std::vector<int>> input = sedova_o_min_of_vector_elements_seq::GetRandomMatrix(rows, columns, min, max);

  int index = (static_cast<int>(gen() % (rows * columns)));
  input[index / columns][index / rows] = ref;

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> task_data_seq = std::make_shared<ppc::core::TaskData>();
  for (unsigned int i = 0; i < input.size(); i++) {
    task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(input[i].data()));
    task_data_seq->inputs_count.emplace_back(rows);
    task_data_seq->inputs_count.emplace_back(columns);

    task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(output.data()));
    task_data_seq->outputs_count.emplace_back(output.size());
  }

  // Create Task
  sedova_o_min_of_vector_elements_seq::TestTaskSequential test_task_sequential(task_data_seq);
  ASSERT_EQ(test_task_sequential.ValidationImpl(), true);
  test_task_sequential.PreProcessingImpl();
  test_task_sequential.RunImpl();
  test_task_sequential.PostProcessingImpl();

  ASSERT_EQ(ref, output[0]);
}

TEST(sedova_o_min_of_vector_elements_seq, test_100x100) {
  std::random_device dev;
  std::mt19937 gen(dev());

  const int rows = 100;
  const int columns = 100;
  const int min = -500;
  const int max = 500;
  int ref = INT_MIN;

  // Create data
  std::vector<int> output(1, INT_MAX);
  std::vector<std::vector<int>> input = sedova_o_min_of_vector_elements_seq::GetRandomMatrix(rows, columns, min, max);

  int index = (static_cast<int>(gen() % (rows * columns)));
  input[index / columns][index / rows] = ref;

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> task_data_seq = std::make_shared<ppc::core::TaskData>();
  for (unsigned int i = 0; i < input.size(); i++) {
    task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(input[i].data()));
    task_data_seq->inputs_count.emplace_back(rows);
    task_data_seq->inputs_count.emplace_back(columns);

    task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(output.data()));
    task_data_seq->outputs_count.emplace_back(output.size());
  }

  // Create Task
  sedova_o_min_of_vector_elements_seq::TestTaskSequential test_task_sequential(task_data_seq);
  ASSERT_EQ(test_task_sequential.ValidationImpl(), true);
  test_task_sequential.PreProcessingImpl();
  test_task_sequential.RunImpl();
  test_task_sequential.PostProcessingImpl();

  ASSERT_EQ(ref, output[0]);
}

TEST(sedova_o_min_of_vector_elements_seq, test_0x0) {
  std::random_device dev;
  std::mt19937 gen(dev());

  const int rows = 0;
  const int columns = 0;
  const int min = -500;
  const int max = 500;

  // Create data
  std::vector<int> output(1, INT_MAX);
  std::vector<std::vector<int>> input = sedova_o_min_of_vector_elements_seq::GetRandomMatrix(rows, columns, min, max);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> task_data_seq = std::make_shared<ppc::core::TaskData>();

  if (!input.empty()) {
    for (unsigned int i = 0; i < input.size(); i++) {
      task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(input[i].data()));
      task_data_seq->inputs_count.emplace_back(input[i].size());

      task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(output.data()));
      task_data_seq->outputs_count.emplace_back(output.size());
    }
  } else {
    task_data_seq->inputs_count.emplace_back(0);
    task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(output.data()));
    task_data_seq->outputs_count.emplace_back(output.size());
  }

  // Create Task
  sedova_o_min_of_vector_elements_seq::TestTaskSequential test_task_sequential(task_data_seq);
  ASSERT_EQ(test_task_sequential.ValidationImpl(), false);
}

TEST(sedova_o_min_of_vector_elements_seq, test_1x1) {
  std::random_device dev;
  std::mt19937 gen(dev());

  const int rows = 1;
  const int columns = 1;
  const int min = -500;
  const int max = 500;
  int ref = INT_MIN;

  // Create data
  std::vector<int> output(1, INT_MAX);
  std::vector<std::vector<int>> input = sedova_o_min_of_vector_elements_seq::GetRandomMatrix(rows, columns, min, max);

  int index = (static_cast<int>(gen() % (rows * columns)));
  input[index / columns][index / rows] = ref;

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> task_data_seq = std::make_shared<ppc::core::TaskData>();
  for (unsigned int i = 0; i < input.size(); i++) {
    task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(input[i].data()));
    task_data_seq->inputs_count.emplace_back(rows);
    task_data_seq->inputs_count.emplace_back(columns);

    task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(output.data()));
    task_data_seq->outputs_count.emplace_back(output.size());
  }

  // Create Task
  sedova_o_min_of_vector_elements_seq::TestTaskSequential test_task_sequential(task_data_seq);
  ASSERT_EQ(test_task_sequential.ValidationImpl(), true);
  test_task_sequential.PreProcessingImpl();
  test_task_sequential.RunImpl();
  test_task_sequential.PostProcessingImpl();

  ASSERT_EQ(ref, output[0]);
}