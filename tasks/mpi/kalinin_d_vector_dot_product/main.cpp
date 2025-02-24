#include <gtest/gtest.h>

#include <boost/mpi/communicator.hpp>
#include <cstdint>
#include <ctime>
#include <memory>
#include <random>
#include <vector>

#include "core/task/include/task.hpp"
#include "mpi/kalinin_d_vector_dot_product/include/ops_mpi.hpp"

namespace {
int offset = 0;
}  // namespace

namespace {
std::vector<int> CreateRandomVector(int v_size) {
  std::vector<int> vec(v_size);
  std::mt19937 gen;
  gen.seed((unsigned)time(nullptr) + ++offset);
  for (int i = 0; i < v_size; i++) {
    vec[i] = static_cast<int>(gen() % 100);
  }
  return vec;
}
}  // namespace

namespace {
std::shared_ptr<ppc::core::TaskData> CreateTaskData(const std::vector<int>& v1, const std::vector<int>& v2,
                                                    std::vector<int32_t>& res) {
  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(const_cast<int*>(v1.data())));
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(const_cast<int*>(v2.data())));
  task_data->inputs_count.emplace_back(v1.size());
  task_data->inputs_count.emplace_back(v2.size());
  task_data->outputs.emplace_back(reinterpret_cast<uint8_t*>(res.data()));
  task_data->outputs_count.emplace_back(res.size());
  return task_data;
}
}  // namespace

namespace {
void RunTest(const std::vector<int>& v1, const std::vector<int>& v2, std::vector<int32_t>& res,
             boost::mpi::communicator& world) {
  auto task_data_mpi = CreateTaskData(v1, v2, res);
  kalinin_d_vector_dot_product_mpi::TestMPITaskParallel test_task_mpi(task_data_mpi);
  ASSERT_EQ(test_task_mpi.Validation(), true);
  test_task_mpi.PreProcessing();
  test_task_mpi.Run();
  test_task_mpi.PostProcessing();

  if (world.rank() == 0) {
    std::vector<int32_t> reference_res(1, 0);
    auto task_data_seq = CreateTaskData(v1, v2, reference_res);
    kalinin_d_vector_dot_product_mpi::TestMPITaskSequential test_task_sequential(task_data_seq);
    ASSERT_EQ(test_task_sequential.Validation(), true);
    test_task_sequential.PreProcessing();
    test_task_sequential.Run();
    test_task_sequential.PostProcessing();
    ASSERT_EQ(reference_res[0], res[0]);
    ASSERT_EQ(kalinin_d_vector_dot_product_mpi::VectorDotProduct(v1, v2), res[0]);
  }
}
}  // namespace

TEST(kalinin_d_vector_dot_product_mpi, can_scalar_multiply_vec_size_300) {
  boost::mpi::communicator world;
  std::vector<int32_t> res(1, 0);
  if (world.rank() == 0) {
    const int k_count_size_vector = 300;
    std::vector<int> v1 = CreateRandomVector(k_count_size_vector);
    std::vector<int> v2 = CreateRandomVector(k_count_size_vector);
    RunTest(v1, v2, res, world);
  }
}

TEST(kalinin_d_vector_dot_product_mpi, check_vectors_not_equal) {
  boost::mpi::communicator world;
  std::vector<int32_t> res(1, 0);
  if (world.rank() == 0) {
    const int k_count_size_vector = 120;
    std::vector<int> v1 = CreateRandomVector(k_count_size_vector);
    std::vector<int> v2 = CreateRandomVector(k_count_size_vector + 5);
    auto task_data_mpi = CreateTaskData(v1, v2, res);
    kalinin_d_vector_dot_product_mpi::TestMPITaskParallel test_task_mpi(task_data_mpi);
    ASSERT_EQ(test_task_mpi.Validation(), false);
  }
}

TEST(kalinin_d_vector_dot_product_mpi, check_vectors_equal_true) {
  boost::mpi::communicator world;
  std::vector<int32_t> res(1, 0);
  if (world.rank() == 0) {
    const int k_count_size_vector = 120;
    std::vector<int> v1 = CreateRandomVector(k_count_size_vector);
    std::vector<int> v2 = CreateRandomVector(k_count_size_vector);
    auto task_data_mpi = CreateTaskData(v1, v2, res);
    kalinin_d_vector_dot_product_mpi::TestMPITaskParallel test_task_mpi(task_data_mpi);
    ASSERT_EQ(test_task_mpi.Validation(), true);
  }
}

TEST(kalinin_d_vector_dot_product_mpi, check_mpi_VectorDotProduct_right) {
  std::vector<int> v1 = {1, 2, 5};
  std::vector<int> v2 = {4, 7, 8};
  ASSERT_EQ(58, kalinin_d_vector_dot_product_mpi::VectorDotProduct(v1, v2));
}

TEST(kalinin_d_vector_dot_product_mpi, check_mpi_Run_right_size_5) {
  boost::mpi::communicator world;
  std::vector<int32_t> res(1, 0);
  if (world.rank() == 0) {
    std::vector<int> v1 = {1, 2, 5, 6, 3};
    std::vector<int> v2 = {4, 7, 8, 9, 5};
    RunTest(v1, v2, res, world);
  }
}

TEST(kalinin_d_vector_dot_product_mpi, check_mpi_Run_right_size_3) {
  boost::mpi::communicator world;
  std::vector<int32_t> res(1, 0);
  if (world.rank() == 0) {
    std::vector<int> v1 = {1, 2, 5};
    std::vector<int> v2 = {4, 7, 8};
    RunTest(v1, v2, res, world);
  }
}

TEST(kalinin_d_vector_dot_product_mpi, check_mpi_Run_right_size_7) {
  boost::mpi::communicator world;
  std::vector<int32_t> res(1, 0);
  if (world.rank() == 0) {
    std::vector<int> v1 = {1, 2, 5, 14, 21, 16, 11};
    std::vector<int> v2 = {4, 7, 8, 12, 31, 25, 9};
    RunTest(v1, v2, res, world);
  }
}

TEST(kalinin_d_vector_dot_product_mpi, check_mpi_Run_right_empty) {
  boost::mpi::communicator world;
  std::vector<int32_t> res(1, 0);
  if (world.rank() == 0) {
    std::vector<int> v1 = {0, 0, 0};
    std::vector<int> v2 = {0, 0, 0};
    RunTest(v1, v2, res, world);
  }
}

TEST(kalinin_d_vector_dot_product_mpi, can_scalar_multiply_vec_size_50) {
  boost::mpi::communicator world;
  std::vector<int32_t> res(1, 0);
  if (world.rank() == 0) {
    const int k_count_size_vector = 50;
    std::vector<int> v1 = CreateRandomVector(k_count_size_vector);
    std::vector<int> v2 = CreateRandomVector(k_count_size_vector);
    RunTest(v1, v2, res, world);
  }
}

TEST(kalinin_d_vector_dot_product_mpi, can_scalar_multiply_vec_size_75) {
  boost::mpi::communicator world;
  std::vector<int32_t> res(1, 0);
  if (world.rank() == 0) {
    const int k_count_size_vector = 75;
    std::vector<int> v1 = CreateRandomVector(k_count_size_vector);
    std::vector<int> v2 = CreateRandomVector(k_count_size_vector);
    RunTest(v1, v2, res, world);
  }
}

TEST(kalinin_d_vector_dot_product_mpi, can_scalar_multiply_vec_size_150) {
  boost::mpi::communicator world;
  std::vector<int32_t> res(1, 0);
  if (world.rank() == 0) {
    const int k_count_size_vector = 150;
    std::vector<int> v1 = CreateRandomVector(k_count_size_vector);
    std::vector<int> v2 = CreateRandomVector(k_count_size_vector);
    RunTest(v1, v2, res, world);
  }
}

TEST(kalinin_d_vector_dot_product_mpi, can_scalar_multiply_vec_size_200) {
  boost::mpi::communicator world;
  std::vector<int32_t> res(1, 0);
  if (world.rank() == 0) {
    const int k_count_size_vector = 200;
    std::vector<int> v1 = CreateRandomVector(k_count_size_vector);
    std::vector<int> v2 = CreateRandomVector(k_count_size_vector);
    RunTest(v1, v2, res, world);
  }
}

TEST(kalinin_d_vector_dot_product_mpi, can_scalar_multiply_vec_size_250) {
  boost::mpi::communicator world;
  std::vector<int32_t> res(1, 0);
  if (world.rank() == 0) {
    const int k_count_size_vector = 250;
    std::vector<int> v1 = CreateRandomVector(k_count_size_vector);
    std::vector<int> v2 = CreateRandomVector(k_count_size_vector);
    RunTest(v1, v2, res, world);
  }
}