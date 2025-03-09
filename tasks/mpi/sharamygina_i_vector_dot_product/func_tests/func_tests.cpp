#include <gtest/gtest.h>

#include <boost/mpi/collectives.hpp>
#include <boost/mpi/communicator.hpp>
#include <cstdint>
#include <memory>
#include <random>
#include <vector>

#include "core/task/include/task.hpp"
#include "mpi/sharamygina_i_vector_dot_product/include/ops_mpi.h"

namespace sharamygina_i_vector_dot_product_mpi {
namespace {
int Resulting(const std::vector<int> &v1, const std::vector<int> &v2) {
  int res = 0;
  for (unsigned int i = 0; i < v1.size(); ++i) {
    res += v1[i] * v2[i];
  }
  return res;
}
std::vector<int> GetVector(unsigned int size) {
  std::random_device dev;
  std::mt19937 gen(dev());
  std::vector<int> v(size);
  for (unsigned int i = 0; i < size; i++) {
    v[i] = static_cast<int>((gen() % 320) - (gen() % 97));
  }
  return v;
}
}  // namespace
}  // namespace sharamygina_i_vector_dot_product_mpi

TEST(sharamygina_i_vector_dot_product_mpi, SampleVecTest) {
  boost::mpi::communicator world;
  unsigned int lenght = 12;

  std::shared_ptr<ppc::core::TaskData> task_data = std::make_shared<ppc::core::TaskData>();

  std::vector<int> received_res(1);
  int expected_res = 90;
  std::vector<int> v1 = {1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4};
  std::vector<int> v2 = {1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4};

  if (world.rank() == 0) {
    task_data->inputs_count.emplace_back(lenght);
    task_data->inputs_count.emplace_back(lenght);

    task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(v1.data()));
    task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(v2.data()));
    task_data->outputs.emplace_back(reinterpret_cast<uint8_t *>(received_res.data()));
    task_data->outputs_count.emplace_back(received_res.size());
  }

  sharamygina_i_vector_dot_product_mpi::VectorDotProductMpi test_task(task_data);
  ASSERT_TRUE(test_task.ValidationImpl());
  ASSERT_TRUE(test_task.PreProcessingImpl());
  ASSERT_TRUE(test_task.RunImpl());
  ASSERT_TRUE(test_task.PostProcessingImpl());

  if (world.rank() == 0) {
    ASSERT_EQ(received_res[0], expected_res);
  }
}

TEST(sharamygina_i_vector_dot_product_mpi, BigVecTest1) {
  boost::mpi::communicator world;
  unsigned int lenght = 3000;

  std::shared_ptr<ppc::core::TaskData> task_data = std::make_shared<ppc::core::TaskData>();

  std::vector<int> received_res(1);
  std::vector<int> v1 = sharamygina_i_vector_dot_product_mpi::GetVector(lenght);
  std::vector<int> v2 = sharamygina_i_vector_dot_product_mpi::GetVector(lenght);
  int expected_res = sharamygina_i_vector_dot_product_mpi::Resulting(v1, v2);

  if (world.rank() == 0) {
    task_data->inputs_count.emplace_back(lenght);
    task_data->inputs_count.emplace_back(lenght);

    task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(v1.data()));
    task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(v2.data()));
    task_data->outputs.emplace_back(reinterpret_cast<uint8_t *>(received_res.data()));
    task_data->outputs_count.emplace_back(received_res.size());
  }

  sharamygina_i_vector_dot_product_mpi::VectorDotProductMpi test_task(task_data);
  ASSERT_TRUE(test_task.ValidationImpl());
  ASSERT_TRUE(test_task.PreProcessingImpl());
  ASSERT_TRUE(test_task.RunImpl());
  ASSERT_TRUE(test_task.PostProcessingImpl());

  if (world.rank() == 0) {
    ASSERT_EQ(received_res[0], expected_res);
  }
}

TEST(sharamygina_i_vector_dot_product_mpi, BigVecTest2) {
  boost::mpi::communicator world;
  unsigned int lenght = 6000;

  std::shared_ptr<ppc::core::TaskData> task_data = std::make_shared<ppc::core::TaskData>();

  std::vector<int> received_res(1);
  std::vector<int> v1 = sharamygina_i_vector_dot_product_mpi::GetVector(lenght);
  std::vector<int> v2 = sharamygina_i_vector_dot_product_mpi::GetVector(lenght);
  int expected_res = sharamygina_i_vector_dot_product_mpi::Resulting(v1, v2);

  if (world.rank() == 0) {
    task_data->inputs_count.emplace_back(lenght);
    task_data->inputs_count.emplace_back(lenght);

    task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(v1.data()));
    task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(v2.data()));
    task_data->outputs.emplace_back(reinterpret_cast<uint8_t *>(received_res.data()));
    task_data->outputs_count.emplace_back(received_res.size());
  }

  sharamygina_i_vector_dot_product_mpi::VectorDotProductMpi test_task(task_data);
  ASSERT_TRUE(test_task.ValidationImpl());
  ASSERT_TRUE(test_task.PreProcessingImpl());
  ASSERT_TRUE(test_task.RunImpl());
  ASSERT_TRUE(test_task.PostProcessingImpl());

  if (world.rank() == 0) {
    ASSERT_EQ(received_res[0], expected_res);
  }
}

TEST(sharamygina_i_vector_dot_product_mpi, BigVecTest3) {
  boost::mpi::communicator world;
  unsigned int lenght = 9000;

  std::shared_ptr<ppc::core::TaskData> task_data = std::make_shared<ppc::core::TaskData>();

  std::vector<int> received_res(1);
  std::vector<int> v1 = sharamygina_i_vector_dot_product_mpi::GetVector(lenght);
  std::vector<int> v2 = sharamygina_i_vector_dot_product_mpi::GetVector(lenght);
  int expected_res = sharamygina_i_vector_dot_product_mpi::Resulting(v1, v2);

  if (world.rank() == 0) {
    task_data->inputs_count.emplace_back(lenght);
    task_data->inputs_count.emplace_back(lenght);

    task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(v1.data()));
    task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(v2.data()));
    task_data->outputs.emplace_back(reinterpret_cast<uint8_t *>(received_res.data()));
    task_data->outputs_count.emplace_back(received_res.size());
  }

  sharamygina_i_vector_dot_product_mpi::VectorDotProductMpi test_task(task_data);
  ASSERT_TRUE(test_task.ValidationImpl());
  ASSERT_TRUE(test_task.PreProcessingImpl());
  ASSERT_TRUE(test_task.RunImpl());
  ASSERT_TRUE(test_task.PostProcessingImpl());

  if (world.rank() == 0) {
    ASSERT_EQ(received_res[0], expected_res);
  }
}

TEST(sharamygina_i_vector_dot_product_mpi, EmptyVecValidationTest) {
  boost::mpi::communicator world;
  unsigned int lenght = 200;

  std::shared_ptr<ppc::core::TaskData> task_data = std::make_shared<ppc::core::TaskData>();

  std::vector<int> received_res(1);

  if (world.rank() == 0) {
    task_data->inputs_count.emplace_back(lenght);
    task_data->inputs_count.emplace_back(lenght);
    task_data->outputs.emplace_back(reinterpret_cast<uint8_t *>(received_res.data()));
    task_data->outputs_count.emplace_back(received_res.size());
  }

  sharamygina_i_vector_dot_product_mpi::VectorDotProductMpi test_task(task_data);
  if (world.rank() == 0) {
    ASSERT_FALSE(test_task.ValidationImpl());
  }
}

TEST(sharamygina_i_vector_dot_product_mpi, DifferentSizeValidationTest) {
  boost::mpi::communicator world;
  unsigned int lenght = 200;

  std::shared_ptr<ppc::core::TaskData> task_data = std::make_shared<ppc::core::TaskData>();

  std::vector<int> received_res(1);
  std::vector<int> v1(lenght);
  std::vector<int> v2(lenght);

  if (world.rank() == 0) {
    task_data->inputs_count.emplace_back(lenght);
    task_data->inputs_count.emplace_back(lenght - 100);

    task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(v1.data()));
    task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(v2.data()));
    task_data->outputs.emplace_back(reinterpret_cast<uint8_t *>(received_res.data()));
    task_data->outputs_count.emplace_back(received_res.size());
  }

  sharamygina_i_vector_dot_product_mpi::VectorDotProductMpi test_task(task_data);
  if (world.rank() == 0) {
    ASSERT_FALSE(test_task.ValidationImpl());
  }
}

TEST(sharamygina_i_vector_dot_product_mpi, EmptyOutputCountValidationTest) {
  boost::mpi::communicator world;
  unsigned int lenght = 200;

  std::shared_ptr<ppc::core::TaskData> task_data = std::make_shared<ppc::core::TaskData>();

  std::vector<int> v1(lenght);
  std::vector<int> v2(lenght);

  if (world.rank() == 0) {
    task_data->inputs_count.emplace_back(lenght);
    task_data->inputs_count.emplace_back(lenght);

    task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(v1.data()));
    task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(v2.data()));
    task_data->outputs_count.emplace_back(1);
  }

  sharamygina_i_vector_dot_product_mpi::VectorDotProductMpi test_task(task_data);
  if (world.rank() == 0) {
    ASSERT_FALSE(test_task.ValidationImpl());
  }
}