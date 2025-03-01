#include <gtest/gtest.h>

#include <boost/mpi/collectives.hpp>
#include <boost/mpi/communicator.hpp>
#include <random>
#include <vector>

#include "mpi/sharamygina_i_vector_dot_product/include/ops_mpi.h"

namespace sharamygina_i_vector_dot_product_mpi {
namespace {
int resulting(const std::vector<int> &v1, const std::vector<int> &v2) {
  int res = 0;
  for (size_t i = 0; i < v1.size(); ++i) {
    res += v1[i] * v2[i];
  }
  return res;
}
std::vector<int> GetVector(int size) {
  std::random_device dev;
  std::mt19937 gen(dev());
  std::vector<int> v(size);
  for (int i = 0; i < size; i++) {
    v[i] = gen() % 320 + gen() % 11;
  }
  return v;
}
}  // namespace
}  // namespace sharamygina_i_vector_dot_product_mpi

TEST(sharamygina_i_vector_dot_product_mpi, SampleVecTest) {
  boost::mpi::communicator world;
  int size1 = 12;
  int size2 = 12;

  std::shared_ptr<ppc::core::TaskData> taskData = std::make_shared<ppc::core::TaskData>();

  std::vector<int> received_res(1);
  int expected_res = 90;
  std::vector<int> v1 = {1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4};
  std::vector<int> v2 = {1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4};

  if (world.rank() == 0) {
    taskData->inputs_count.emplace_back(size1);
    taskData->inputs_count.emplace_back(size2);

    taskData->inputs.emplace_back(reinterpret_cast<uint8_t *>(v1.data()));
    taskData->inputs.emplace_back(reinterpret_cast<uint8_t *>(v2.data()));
    taskData->outputs.emplace_back(reinterpret_cast<uint8_t *>(received_res.data()));
    taskData->outputs_count.emplace_back(received_res.size());
  }

  sharamygina_i_vector_dot_product_mpi::vector_dot_product_mpi testTask(taskData);
  ASSERT_TRUE(testTask.ValidationImpl());
  ASSERT_TRUE(testTask.PreProcessingImpl());
  ASSERT_TRUE(testTask.RunImpl());
  ASSERT_TRUE(testTask.PostProcessingImpl());

  if (world.rank() == 0) {
    ASSERT_EQ(received_res[0], expected_res);
  }
}

TEST(sharamygina_i_vector_dot_product_mpi, BigVecTest) {
  boost::mpi::communicator world;
  int size1 = 3000;
  int size2 = 3000;

  std::shared_ptr<ppc::core::TaskData> taskData = std::make_shared<ppc::core::TaskData>();

  std::vector<int> received_res(1);
  std::vector<int> v1 = sharamygina_i_vector_dot_product_mpi::GetVector(size1);
  std::vector<int> v2 = sharamygina_i_vector_dot_product_mpi::GetVector(size2);
  int expected_res = sharamygina_i_vector_dot_product_mpi::resulting(v1, v2);

  if (world.rank() == 0) {
    taskData->inputs_count.emplace_back(size1);
    taskData->inputs_count.emplace_back(size2);

    taskData->inputs.emplace_back(reinterpret_cast<uint8_t *>(v1.data()));
    taskData->inputs.emplace_back(reinterpret_cast<uint8_t *>(v2.data()));
    taskData->outputs.emplace_back(reinterpret_cast<uint8_t *>(received_res.data()));
    taskData->outputs_count.emplace_back(received_res.size());
  }

  sharamygina_i_vector_dot_product_mpi::vector_dot_product_mpi testTask(taskData);
  ASSERT_TRUE(testTask.ValidationImpl());
  ASSERT_TRUE(testTask.PreProcessingImpl());
  ASSERT_TRUE(testTask.RunImpl());
  ASSERT_TRUE(testTask.PostProcessingImpl());

  if (world.rank() == 0) {
    ASSERT_EQ(received_res[0], expected_res);
  }
}

TEST(sharamygina_i_vector_dot_product_mpi, EmptyVecValidationTest) {
  boost::mpi::communicator world;
  int size1 = 200;
  int size2 = 200;

  std::shared_ptr<ppc::core::TaskData> taskData = std::make_shared<ppc::core::TaskData>();

  std::vector<int> received_res(1);

  if (world.rank() == 0) {
    taskData->inputs_count.emplace_back(size1);
    taskData->inputs_count.emplace_back(size2);
    taskData->outputs.emplace_back(reinterpret_cast<uint8_t *>(received_res.data()));
    taskData->outputs_count.emplace_back(received_res.size());
  }

  sharamygina_i_vector_dot_product_mpi::vector_dot_product_mpi testTask(taskData);
  if (world.rank() == 0) ASSERT_FALSE(testTask.ValidationImpl());
}

TEST(sharamygina_i_vector_dot_product_mpi, OneSizeValidationTest) {
  boost::mpi::communicator world;
  int size1 = 200;

  std::shared_ptr<ppc::core::TaskData> taskData = std::make_shared<ppc::core::TaskData>();

  std::vector<int> received_res(1);
  std::vector<int> v1(size1);
  std::vector<int> v2(size1);

  if (world.rank() == 0) {
    taskData->inputs_count.emplace_back(size1);

    taskData->inputs.emplace_back(reinterpret_cast<uint8_t *>(v1.data()));
    taskData->inputs.emplace_back(reinterpret_cast<uint8_t *>(v2.data()));
    taskData->outputs.emplace_back(reinterpret_cast<uint8_t *>(received_res.data()));
    taskData->outputs_count.emplace_back(received_res.size());
  }

  sharamygina_i_vector_dot_product_mpi::vector_dot_product_mpi testTask(taskData);
  if (world.rank() == 0) ASSERT_FALSE(testTask.ValidationImpl());
}

TEST(sharamygina_i_vector_dot_product_mpi, OneVecAndSizeValidationTest) {
  boost::mpi::communicator world;
  int size1 = 200;

  std::shared_ptr<ppc::core::TaskData> taskData = std::make_shared<ppc::core::TaskData>();

  std::vector<int> received_res(1);
  std::vector<int> v1(size1);

  if (world.rank() == 0) {
    taskData->inputs_count.emplace_back(size1);

    taskData->inputs.emplace_back(reinterpret_cast<uint8_t *>(v1.data()));
    taskData->outputs.emplace_back(reinterpret_cast<uint8_t *>(received_res.data()));
    taskData->outputs_count.emplace_back(received_res.size());
  }

  sharamygina_i_vector_dot_product_mpi::vector_dot_product_mpi testTask(taskData);
  if (world.rank() == 0) ASSERT_FALSE(testTask.ValidationImpl());
}

TEST(sharamygina_i_vector_dot_product_mpi, EmptyOutputCountValidationTest) {
  boost::mpi::communicator world;
  int size1 = 200;
  int size2 = 200;

  std::shared_ptr<ppc::core::TaskData> taskData = std::make_shared<ppc::core::TaskData>();

  std::vector<int> v1(size1);
  std::vector<int> v2(size2);

  if (world.rank() == 0) {
    taskData->inputs_count.emplace_back(size1);
    taskData->inputs_count.emplace_back(size2);

    taskData->inputs.emplace_back(reinterpret_cast<uint8_t *>(v1.data()));
    taskData->inputs.emplace_back(reinterpret_cast<uint8_t *>(v2.data()));
    taskData->outputs_count.emplace_back(1);
  }

  sharamygina_i_vector_dot_product_mpi::vector_dot_product_mpi testTask(taskData);
  if (world.rank() == 0) ASSERT_FALSE(testTask.ValidationImpl());
}