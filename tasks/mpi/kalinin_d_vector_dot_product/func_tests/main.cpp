
#include <gtest/gtest.h>

#include <boost/mpi/communicator.hpp>
#include <cstddef>
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

TEST(kalinin_d_vector_dot_product_mpi, can_scalar_multiply_vec_size_125) {
  boost::mpi::communicator world;
  std::vector<std::vector<int>> global_vec;
  std::vector<int32_t> res(1, 0);
  // Create task_data
  auto task_data_mpi = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    const int count_size_vector = 125;
    std::vector<int> v1 = CreateRandomVector(count_size_vector);
    std::vector<int> v2 = CreateRandomVector(count_size_vector);

    global_vec = {v1, v2};
    for (size_t i = 0; i < global_vec.size(); i++) {
      task_data_mpi->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vec[i].data()));
    }
    task_data_mpi->inputs_count.emplace_back(global_vec[0].size());
    task_data_mpi->inputs_count.emplace_back(global_vec[1].size());
    task_data_mpi->outputs.emplace_back(reinterpret_cast<uint8_t*>(res.data()));
    task_data_mpi->outputs_count.emplace_back(res.size());
  }

  kalinin_d_vector_dot_product_mpi::TestMPITaskParallel test_task_mpi(task_data_mpi);
  test_task_mpi.Validation();
  test_task_mpi.PreProcessing();
  test_task_mpi.Run();
  test_task_mpi.PostProcessing();

  if (world.rank() == 0) {
    // Create data
    std::vector<int32_t> reference_res(1, 0);

    // Create TaskData
    auto task_data_seq = std::make_shared<ppc::core::TaskData>();
    for (size_t i = 0; i < global_vec.size(); i++) {
      task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vec[i].data()));
    }
    task_data_seq->inputs_count.emplace_back(global_vec[0].size());
    task_data_seq->inputs_count.emplace_back(global_vec[1].size());
    task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t*>(reference_res.data()));
    task_data_seq->outputs_count.emplace_back(reference_res.size());

    // Create Task
    kalinin_d_vector_dot_product_mpi::TestMPITaskSequential test_task_sequential(task_data_seq);
    test_task_sequential.Validation();
    test_task_sequential.PreProcessing();
    test_task_sequential.Run();
    test_task_sequential.PostProcessing();
    ASSERT_EQ(reference_res[0], res[0]);
  }
}

TEST(kalinin_d_vector_dot_product_mpi, can_scalar_multiply_vec_size_300) {
  boost::mpi::communicator world;
  std::vector<std::vector<int>> global_vec;
  std::vector<int32_t> res(1, 0);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> task_data_mpi = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    const int count_size_vector = 300;
    std::vector<int> v1 = CreateRandomVector(count_size_vector);
    std::vector<int> v2 = CreateRandomVector(count_size_vector);

    global_vec = {v1, v2};
    for (size_t i = 0; i < global_vec.size(); i++) {
      task_data_mpi->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vec[i].data()));
    }
    task_data_mpi->inputs_count.emplace_back(global_vec[0].size());
    task_data_mpi->inputs_count.emplace_back(global_vec[1].size());
    task_data_mpi->outputs.emplace_back(reinterpret_cast<uint8_t*>(res.data()));
    task_data_mpi->outputs_count.emplace_back(res.size());
  }

  kalinin_d_vector_dot_product_mpi::TestMPITaskParallel test_task_mpi(task_data_mpi);
  test_task_mpi.Validation();
  test_task_mpi.PreProcessing();
  test_task_mpi.Run();
  test_task_mpi.PostProcessing();

  if (world.rank() == 0) {
    // Create data
    std::vector<int32_t> reference_res(1, 0);

    // Create TaskData
    std::shared_ptr<ppc::core::TaskData> task_data_seq = std::make_shared<ppc::core::TaskData>();
    for (size_t i = 0; i < global_vec.size(); i++) {
      task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vec[i].data()));
    }
    task_data_seq->inputs_count.emplace_back(global_vec[0].size());
    task_data_seq->inputs_count.emplace_back(global_vec[1].size());
    task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t*>(reference_res.data()));
    task_data_seq->outputs_count.emplace_back(reference_res.size());

    // Create Task
    kalinin_d_vector_dot_product_mpi::TestMPITaskSequential test_task_sequential(task_data_seq);
    test_task_sequential.Validation();
    test_task_sequential.PreProcessing();
    test_task_sequential.Run();
    test_task_sequential.PostProcessing();
    ASSERT_EQ(reference_res[0], res[0]);
  }
}

TEST(kalinin_d_vector_dot_product_mpi, check_vectors_not_equal) {
  boost::mpi::communicator world;
  std::vector<std::vector<int>> global_vec;
  std::vector<int32_t> res(1, 0);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> task_data_mpi = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    const int count_size_vector = 120;
    std::vector<int> v1 = CreateRandomVector(count_size_vector);
    std::vector<int> v2 = CreateRandomVector(count_size_vector + 5);

    global_vec = {v1, v2};
    for (size_t i = 0; i < global_vec.size(); i++) {
      task_data_mpi->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vec[i].data()));
    }
    task_data_mpi->inputs_count.emplace_back(global_vec[0].size());
    task_data_mpi->inputs_count.emplace_back(global_vec[1].size());
    task_data_mpi->outputs.emplace_back(reinterpret_cast<uint8_t*>(res.data()));
    task_data_mpi->outputs_count.emplace_back(res.size());
    kalinin_d_vector_dot_product_mpi::TestMPITaskParallel test_task_mpi(task_data_mpi);
    ASSERT_EQ(test_task_mpi.Validation(), false);
  }
  // Create Task
}

TEST(kalinin_d_vector_dot_product_mpi, check_vectors_equal_true) {
  boost::mpi::communicator world;
  std::vector<std::vector<int>> global_vec;
  std::vector<int32_t> res(1, 0);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> task_data_mpi = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    const int count_size_vector = 120;
    std::vector<int> v1 = CreateRandomVector(count_size_vector);
    std::vector<int> v2 = CreateRandomVector(count_size_vector);

    global_vec = {v1, v2};
    for (size_t i = 0; i < global_vec.size(); i++) {
      task_data_mpi->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vec[i].data()));
    }
    task_data_mpi->inputs_count.emplace_back(global_vec[0].size());
    task_data_mpi->inputs_count.emplace_back(global_vec[1].size());
    task_data_mpi->outputs.emplace_back(reinterpret_cast<uint8_t*>(res.data()));
    task_data_mpi->outputs_count.emplace_back(res.size());
    kalinin_d_vector_dot_product_mpi::TestMPITaskParallel test_task_mpi(task_data_mpi);
    ASSERT_EQ(test_task_mpi.Validation(), true);
  }
  // Create Task
}

TEST(kalinin_d_vector_dot_product_mpi, check_mpi_vectorDotProduct_right) {
  // Create data
  std::vector<int> v1 = {1, 2, 5};
  std::vector<int> v2 = {4, 7, 8};
  ASSERT_EQ(58, kalinin_d_vector_dot_product_mpi::VectorDotProduct(v1, v2));
}

TEST(kalinin_d_vector_dot_product_mpi, check_mpi_Run_right_size_5) {
  boost::mpi::communicator world;
  std::vector<std::vector<int>> global_vec;
  std::vector<int32_t> res(1, 0);
  std::vector<int> v1 = {1, 2, 5, 6, 3};
  std::vector<int> v2 = {4, 7, 8, 9, 5};
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> task_data_mpi = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    global_vec = {v1, v2};
    for (size_t i = 0; i < global_vec.size(); i++) {
      task_data_mpi->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vec[i].data()));
    }
    task_data_mpi->inputs_count.emplace_back(global_vec[0].size());
    task_data_mpi->inputs_count.emplace_back(global_vec[1].size());
    task_data_mpi->outputs.emplace_back(reinterpret_cast<uint8_t*>(res.data()));
    task_data_mpi->outputs_count.emplace_back(res.size());
  }
  kalinin_d_vector_dot_product_mpi::TestMPITaskParallel test_task_mpi(task_data_mpi);
  test_task_mpi.Validation();
  test_task_mpi.PreProcessing();
  test_task_mpi.Run();
  test_task_mpi.PostProcessing();
  if (world.rank() == 0) {
    ASSERT_EQ(kalinin_d_vector_dot_product_mpi::VectorDotProduct(v1, v2), res[0]);
  }
}

TEST(kalinin_d_vector_dot_product_mpi, check_mpi_Run_right_size_3) {
  boost::mpi::communicator world;
  std::vector<std::vector<int>> global_vec;
  std::vector<int32_t> res(1, 0);
  std::vector<int> v1 = {1, 2, 5};
  std::vector<int> v2 = {4, 7, 8};
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> task_data_mpi = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    global_vec = {v1, v2};
    for (size_t i = 0; i < global_vec.size(); i++) {
      task_data_mpi->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vec[i].data()));
    }
    task_data_mpi->inputs_count.emplace_back(global_vec[0].size());
    task_data_mpi->inputs_count.emplace_back(global_vec[1].size());
    task_data_mpi->outputs.emplace_back(reinterpret_cast<uint8_t*>(res.data()));
    task_data_mpi->outputs_count.emplace_back(res.size());
  }
  kalinin_d_vector_dot_product_mpi::TestMPITaskParallel test_task_mpi(task_data_mpi);
  test_task_mpi.Validation();
  test_task_mpi.PreProcessing();
  test_task_mpi.Run();
  test_task_mpi.PostProcessing();
  if (world.rank() == 0) {
    ASSERT_EQ(58, res[0]);
  }
}

TEST(kalinin_d_vector_dot_product_mpi, check_mpi_Run_right_size_7) {
  boost::mpi::communicator world;
  std::vector<std::vector<int>> global_vec;
  std::vector<int32_t> res(1, 0);
  std::vector<int> v1 = {1, 2, 5, 14, 21, 16, 11};
  std::vector<int> v2 = {4, 7, 8, 12, 31, 25, 9};
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> task_data_mpi = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    global_vec = {v1, v2};
    for (size_t i = 0; i < global_vec.size(); i++) {
      task_data_mpi->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vec[i].data()));
    }
    task_data_mpi->inputs_count.emplace_back(global_vec[0].size());
    task_data_mpi->inputs_count.emplace_back(global_vec[1].size());
    task_data_mpi->outputs.emplace_back(reinterpret_cast<uint8_t*>(res.data()));
    task_data_mpi->outputs_count.emplace_back(res.size());
  }
  kalinin_d_vector_dot_product_mpi::TestMPITaskParallel test_task_mpi(task_data_mpi);
  test_task_mpi.Validation();
  test_task_mpi.PreProcessing();
  test_task_mpi.Run();
  test_task_mpi.PostProcessing();
  if (world.rank() == 0) {
    ASSERT_EQ(kalinin_d_vector_dot_product_mpi::VectorDotProduct(v1, v2), res[0]);
  }
}

TEST(kalinin_d_vector_dot_product_mpi, check_mpi_Run_right_empty) {
  boost::mpi::communicator world;
  std::vector<std::vector<int>> global_vec;
  std::vector<int32_t> res(1, 0);
  std::vector<int> v1 = {0, 0, 0};
  std::vector<int> v2 = {0, 0, 0};
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> task_data_mpi = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    global_vec = {v1, v2};
    for (size_t i = 0; i < global_vec.size(); i++) {
      task_data_mpi->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vec[i].data()));
    }
    task_data_mpi->inputs_count.emplace_back(global_vec[0].size());
    task_data_mpi->inputs_count.emplace_back(global_vec[1].size());
    task_data_mpi->outputs.emplace_back(reinterpret_cast<uint8_t*>(res.data()));
    task_data_mpi->outputs_count.emplace_back(res.size());
  }
  kalinin_d_vector_dot_product_mpi::TestMPITaskParallel test_task_mpi(task_data_mpi);
  test_task_mpi.Validation();
  test_task_mpi.PreProcessing();
  test_task_mpi.Run();
  test_task_mpi.PostProcessing();
  if (world.rank() == 0) {
    ASSERT_EQ(kalinin_d_vector_dot_product_mpi::VectorDotProduct(v1, v2), res[0]);
  }
}

TEST(kalinin_d_vector_dot_product_mpi, can_scalar_multiply_vec_size_50) {
  boost::mpi::communicator world;
  std::vector<std::vector<int>> global_vec;
  std::vector<int32_t> res(1, 0);
  auto task_data_mpi = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    const int count_size_vector = 50;
    std::vector<int> v1 = CreateRandomVector(count_size_vector);
    std::vector<int> v2 = CreateRandomVector(count_size_vector);

    global_vec = {v1, v2};
    for (size_t i = 0; i < global_vec.size(); i++) {
      task_data_mpi->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vec[i].data()));
    }
    task_data_mpi->inputs_count.emplace_back(global_vec[0].size());
    task_data_mpi->inputs_count.emplace_back(global_vec[1].size());
    task_data_mpi->outputs.emplace_back(reinterpret_cast<uint8_t*>(res.data()));
    task_data_mpi->outputs_count.emplace_back(res.size());
  }

  kalinin_d_vector_dot_product_mpi::TestMPITaskParallel test_task_mpi(task_data_mpi);
  test_task_mpi.Validation();
  test_task_mpi.PreProcessing();
  test_task_mpi.Run();
  test_task_mpi.PostProcessing();

  if (world.rank() == 0) {
    std::vector<int32_t> reference_res(1, 0);
    auto task_data_seq = std::make_shared<ppc::core::TaskData>();
    for (size_t i = 0; i < global_vec.size(); i++) {
      task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vec[i].data()));
    }
    task_data_seq->inputs_count.emplace_back(global_vec[0].size());
    task_data_seq->inputs_count.emplace_back(global_vec[1].size());
    task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t*>(reference_res.data()));
    task_data_seq->outputs_count.emplace_back(reference_res.size());

    kalinin_d_vector_dot_product_mpi::TestMPITaskSequential test_task_sequential(task_data_seq);
    test_task_sequential.Validation();
    test_task_sequential.PreProcessing();
    test_task_sequential.Run();
    test_task_sequential.PostProcessing();
    ASSERT_EQ(reference_res[0], res[0]);
  }
}

TEST(kalinin_d_vector_dot_product_mpi, can_scalar_multiply_vec_size_75) {
  boost::mpi::communicator world;
  std::vector<std::vector<int>> global_vec;
  std::vector<int32_t> res(1, 0);
  auto task_data_mpi = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    const int count_size_vector = 75;
    std::vector<int> v1 = CreateRandomVector(count_size_vector);
    std::vector<int> v2 = CreateRandomVector(count_size_vector);

    global_vec = {v1, v2};
    for (size_t i = 0; i < global_vec.size(); i++) {
      task_data_mpi->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vec[i].data()));
    }
    task_data_mpi->inputs_count.emplace_back(global_vec[0].size());
    task_data_mpi->inputs_count.emplace_back(global_vec[1].size());
    task_data_mpi->outputs.emplace_back(reinterpret_cast<uint8_t*>(res.data()));
    task_data_mpi->outputs_count.emplace_back(res.size());
  }

  kalinin_d_vector_dot_product_mpi::TestMPITaskParallel test_task_mpi(task_data_mpi);
  test_task_mpi.Validation();
  test_task_mpi.PreProcessing();
  test_task_mpi.Run();
  test_task_mpi.PostProcessing();

  if (world.rank() == 0) {
    std::vector<int32_t> reference_res(1, 0);
    auto task_data_seq = std::make_shared<ppc::core::TaskData>();
    for (size_t i = 0; i < global_vec.size(); i++) {
      task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vec[i].data()));
    }
    task_data_seq->inputs_count.emplace_back(global_vec[0].size());
    task_data_seq->inputs_count.emplace_back(global_vec[1].size());
    task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t*>(reference_res.data()));
    task_data_seq->outputs_count.emplace_back(reference_res.size());

    kalinin_d_vector_dot_product_mpi::TestMPITaskSequential test_task_sequential(task_data_seq);
    test_task_sequential.Validation();
    test_task_sequential.PreProcessing();
    test_task_sequential.Run();
    test_task_sequential.PostProcessing();
    ASSERT_EQ(reference_res[0], res[0]);
  }
}

TEST(kalinin_d_vector_dot_product_mpi, can_scalar_multiply_vec_size_150) {
  boost::mpi::communicator world;
  std::vector<std::vector<int>> global_vec;
  std::vector<int32_t> res(1, 0);
  auto task_data_mpi = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    const int count_size_vector = 150;
    std::vector<int> v1 = CreateRandomVector(count_size_vector);
    std::vector<int> v2 = CreateRandomVector(count_size_vector);

    global_vec = {v1, v2};
    for (size_t i = 0; i < global_vec.size(); i++) {
      task_data_mpi->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vec[i].data()));
    }
    task_data_mpi->inputs_count.emplace_back(global_vec[0].size());
    task_data_mpi->inputs_count.emplace_back(global_vec[1].size());
    task_data_mpi->outputs.emplace_back(reinterpret_cast<uint8_t*>(res.data()));
    task_data_mpi->outputs_count.emplace_back(res.size());
  }

  kalinin_d_vector_dot_product_mpi::TestMPITaskParallel test_task_mpi(task_data_mpi);
  test_task_mpi.Validation();
  test_task_mpi.PreProcessing();
  test_task_mpi.Run();
  test_task_mpi.PostProcessing();

  if (world.rank() == 0) {
    std::vector<int32_t> reference_res(1, 0);
    auto task_data_seq = std::make_shared<ppc::core::TaskData>();
    for (size_t i = 0; i < global_vec.size(); i++) {
      task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vec[i].data()));
    }
    task_data_seq->inputs_count.emplace_back(global_vec[0].size());
    task_data_seq->inputs_count.emplace_back(global_vec[1].size());
    task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t*>(reference_res.data()));
    task_data_seq->outputs_count.emplace_back(reference_res.size());

    kalinin_d_vector_dot_product_mpi::TestMPITaskSequential test_task_sequential(task_data_seq);
    test_task_sequential.Validation();
    test_task_sequential.PreProcessing();
    test_task_sequential.Run();
    test_task_sequential.PostProcessing();
    ASSERT_EQ(reference_res[0], res[0]);
  }
}

TEST(kalinin_d_vector_dot_product_mpi, can_scalar_multiply_vec_size_200) {
  boost::mpi::communicator world;
  std::vector<std::vector<int>> global_vec;
  std::vector<int32_t> res(1, 0);
  auto task_data_mpi = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    const int count_size_vector = 200;
    std::vector<int> v1 = CreateRandomVector(count_size_vector);
    std::vector<int> v2 = CreateRandomVector(count_size_vector);

    global_vec = {v1, v2};
    for (size_t i = 0; i < global_vec.size(); i++) {
      task_data_mpi->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vec[i].data()));
    }
    task_data_mpi->inputs_count.emplace_back(global_vec[0].size());
    task_data_mpi->inputs_count.emplace_back(global_vec[1].size());
    task_data_mpi->outputs.emplace_back(reinterpret_cast<uint8_t*>(res.data()));
    task_data_mpi->outputs_count.emplace_back(res.size());
  }

  kalinin_d_vector_dot_product_mpi::TestMPITaskParallel test_task_mpi(task_data_mpi);
  test_task_mpi.Validation();
  test_task_mpi.PreProcessing();
  test_task_mpi.Run();
  test_task_mpi.PostProcessing();

  if (world.rank() == 0) {
    std::vector<int32_t> reference_res(1, 0);
    auto task_data_seq = std::make_shared<ppc::core::TaskData>();
    for (size_t i = 0; i < global_vec.size(); i++) {
      task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vec[i].data()));
    }
    task_data_seq->inputs_count.emplace_back(global_vec[0].size());
    task_data_seq->inputs_count.emplace_back(global_vec[1].size());
    task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t*>(reference_res.data()));
    task_data_seq->outputs_count.emplace_back(reference_res.size());

    kalinin_d_vector_dot_product_mpi::TestMPITaskSequential test_task_sequential(task_data_seq);
    test_task_sequential.Validation();
    test_task_sequential.PreProcessing();
    test_task_sequential.Run();
    test_task_sequential.PostProcessing();
    ASSERT_EQ(reference_res[0], res[0]);
  }
}

TEST(kalinin_d_vector_dot_product_mpi, can_scalar_multiply_vec_size_250) {
  boost::mpi::communicator world;
  std::vector<std::vector<int>> global_vec;
  std::vector<int32_t> res(1, 0);
  auto task_data_mpi = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    const int count_size_vector = 250;
    std::vector<int> v1 = CreateRandomVector(count_size_vector);
    std::vector<int> v2 = CreateRandomVector(count_size_vector);

    global_vec = {v1, v2};
    for (size_t i = 0; i < global_vec.size(); i++) {
      task_data_mpi->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vec[i].data()));
    }
    task_data_mpi->inputs_count.emplace_back(global_vec[0].size());
    task_data_mpi->inputs_count.emplace_back(global_vec[1].size());
    task_data_mpi->outputs.emplace_back(reinterpret_cast<uint8_t*>(res.data()));
    task_data_mpi->outputs_count.emplace_back(res.size());
  }

  kalinin_d_vector_dot_product_mpi::TestMPITaskParallel test_task_mpi(task_data_mpi);
  test_task_mpi.Validation();
  test_task_mpi.PreProcessing();
  test_task_mpi.Run();
  test_task_mpi.PostProcessing();

  if (world.rank() == 0) {
    std::vector<int32_t> reference_res(1, 0);
    auto task_data_seq = std::make_shared<ppc::core::TaskData>();
    for (size_t i = 0; i < global_vec.size(); i++) {
      task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vec[i].data()));
    }
    task_data_seq->inputs_count.emplace_back(global_vec[0].size());
    task_data_seq->inputs_count.emplace_back(global_vec[1].size());
    task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t*>(reference_res.data()));
    task_data_seq->outputs_count.emplace_back(reference_res.size());

    kalinin_d_vector_dot_product_mpi::TestMPITaskSequential test_task_sequential(task_data_seq);
    test_task_sequential.Validation();
    test_task_sequential.PreProcessing();
    test_task_sequential.Run();
    test_task_sequential.PostProcessing();
    ASSERT_EQ(reference_res[0], res[0]);
  }
}