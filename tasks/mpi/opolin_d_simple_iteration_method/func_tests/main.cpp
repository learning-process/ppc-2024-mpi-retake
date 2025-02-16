// Copyright 2023 Nesterov Alexander
#include <gtest/gtest.h>

#include <climits>
#include <memory>
#include <random>
#include <string>
#include <vector>

#include "core/task/include/task.hpp"
#include "core/util/include/util.hpp"
#include "mpi/opolin_d_simple_iteration_method/include/ops_mpi.hpp"

namespace opolin_d_simple_iteration_method_mpi {
namespace {
void generateTestData(size_t size, std::vector<double> &X, std::vector<double> &A, std::vector<double> &b) {
  std::srand(static_cast<unsigned>(std::time(nullptr)));

  X.resize(size);
  for (size_t i = 0; i < size; ++i) {
    X[i] = -10.0 + static_cast<double>(std::rand() % 1000) / 50.0;
  }

  A.resize(size * size, 0.0);
  for (size_t i = 0; i < size; ++i) {
    double sum = 0.0;
    for (size_t j = 0; j < size; ++j) {
      if (i != j) {
        A[i * size + j] = -1.0 + static_cast<double>(std::rand() % 1000) / 500.0;
        sum += std::abs(A[i * size + j]);
      }
    }
    A[i * size + i] = sum + 1.0;
  }
  b.resize(size, 0.0);
  for (size_t i = 0; i < size; ++i) {
    for (size_t j = 0; j < size; ++j) {
      b[i] += A[i * size + j] * X[j];
    }
  }
}
}  // namespace
}  // namespace opolin_d_simple_iteration_method_mpi

TEST(opolin_d_simple_iteration_method_mpi, test_small_system) {
  boost::mpi::communicator world;
  int size = 5;
  double epsilon = 1e-8;
  int maxIters = 10000;

  std::vector<double> x_ref, A, b;

  std::vector<double> x_out(size, 0.0);
  auto task_data_mpi = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    opolin_d_simple_iteration_method_mpi::generateTestData(size, x_ref, A, b);
    task_data_mpi->inputs.emplace_back(reinterpret_cast<uint8_t *>(A.data()));
    task_data_mpi->inputs_count.emplace_back(x_out.size());
    task_data_mpi->inputs.emplace_back(reinterpret_cast<uint8_t *>(b.data()));
    task_data_mpi->inputs.emplace_back(reinterpret_cast<uint8_t *>(&epsilon));
    task_data_mpi->inputs.emplace_back(reinterpret_cast<uint8_t *>(&maxIters));
    task_data_mpi->outputs.emplace_back(reinterpret_cast<uint8_t *>(x_out.data()));
    task_data_mpi->outputs_count.emplace_back(x_out.size());
  }
  opolin_d_simple_iteration_method_mpi::TestTaskMPI test_task_mpi(task_data_mpi);

  ASSERT_EQ(test_task_mpi.ValidationImpl(), true);
  test_task_mpi.PreProcessingImpl();
  test_task_mpi.RunImpl();
  test_task_mpi.PostProcessingImpl();
  if (world.rank() == 0) {
    for (size_t i = 0; i < x_ref.size(); ++i) {
      ASSERT_NEAR(x_ref[i], x_out[i], 1e-3);
    }
  }
}

TEST(opolin_d_simple_iteration_method_mpi, test_big_system) {
  boost::mpi::communicator world;
  int size = 50;
  double epsilon = 1e-8;
  int maxIters = 10000;

  std::vector<double> x_ref, A, b;

  std::vector<double> x_out(size, 0.0);
  auto task_data_mpi = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    opolin_d_simple_iteration_method_mpi::generateTestData(size, x_ref, A, b);
    task_data_mpi->inputs.emplace_back(reinterpret_cast<uint8_t *>(A.data()));
    task_data_mpi->inputs_count.emplace_back(x_out.size());
    task_data_mpi->inputs.emplace_back(reinterpret_cast<uint8_t *>(b.data()));
    task_data_mpi->inputs.emplace_back(reinterpret_cast<uint8_t *>(&epsilon));
    task_data_mpi->inputs.emplace_back(reinterpret_cast<uint8_t *>(&maxIters));
    task_data_mpi->outputs.emplace_back(reinterpret_cast<uint8_t *>(x_out.data()));
    task_data_mpi->outputs_count.emplace_back(x_out.size());
  }
  opolin_d_simple_iteration_method_mpi::TestTaskMPI test_task_mpi(task_data_mpi);

  ASSERT_EQ(test_task_mpi.ValidationImpl(), true);
  test_task_mpi.PreProcessingImpl();
  test_task_mpi.RunImpl();
  test_task_mpi.PostProcessingImpl();
  if (world.rank() == 0) {
    for (size_t i = 0; i < x_ref.size(); ++i) {
      ASSERT_NEAR(x_ref[i], x_out[i], 1e-3);
    }
  }
}

TEST(opolin_d_simple_iteration_method_mpi, test_correct_input) {
  boost::mpi::communicator world;
  int size = 3;
  double epsilon = 1e-8;
  int maxIters = 10000;

  std::vector<double> x_ref, A, b;
  std::vector<double> x_out(size, 0.0);
  auto task_data_mpi = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    A = {4.0, 1.0, 2.0, 1.0, 5.0, 1.0, 2.0, 1.0, 5.0};
    b = {7.0, 7.0, 8.0};
    x_ref = {1.0, 1.0, 1.0};

    task_data_mpi->inputs.emplace_back(reinterpret_cast<uint8_t *>(A.data()));
    task_data_mpi->inputs_count.emplace_back(x_out.size());
    task_data_mpi->inputs.emplace_back(reinterpret_cast<uint8_t *>(b.data()));
    task_data_mpi->inputs.emplace_back(reinterpret_cast<uint8_t *>(&epsilon));
    task_data_mpi->inputs.emplace_back(reinterpret_cast<uint8_t *>(&maxIters));
    task_data_mpi->outputs.emplace_back(reinterpret_cast<uint8_t *>(x_out.data()));
    task_data_mpi->outputs_count.emplace_back(x_out.size());
  }
  opolin_d_simple_iteration_method_mpi::TestTaskMPI test_task_mpi(task_data_mpi);

  ASSERT_EQ(test_task_mpi.ValidationImpl(), true);
  test_task_mpi.PreProcessingImpl();
  test_task_mpi.RunImpl();
  test_task_mpi.PostProcessingImpl();
  if (world.rank() == 0) {
    for (size_t i = 0; i < x_ref.size(); ++i) {
      ASSERT_NEAR(x_ref[i], x_out[i], 1e-3);
    }
  }
}

TEST(opolin_d_simple_iteration_method_mpi, test_no_dominance_matrix) {
  boost::mpi::communicator world;
  int size = 3;
  double epsilon = 1e-8;
  int maxIters = 1000;
  auto task_data_mpi = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    std::vector<double> A = {3.0, 2.0, 4.0, 1.0, 2.0, 4.0, 1.0, 2.0, 3.0};
    std::vector<double> b = {3.0, 2.0, 2.0};

    std::vector<double> x_out(size, 0.0);

    task_data_mpi->inputs.emplace_back(reinterpret_cast<uint8_t *>(A.data()));
    task_data_mpi->inputs_count.emplace_back(x_out.size());
    task_data_mpi->inputs.emplace_back(reinterpret_cast<uint8_t *>(b.data()));
    task_data_mpi->inputs.emplace_back(reinterpret_cast<uint8_t *>(&epsilon));
    task_data_mpi->inputs.emplace_back(reinterpret_cast<uint8_t *>(&maxIters));
    task_data_mpi->outputs.emplace_back(reinterpret_cast<uint8_t *>(x_out.data()));
    task_data_mpi->outputs_count.emplace_back(x_out.size());

    opolin_d_simple_iteration_method_mpi::TestTaskMPI test_task_mpi(task_data_mpi);

    ASSERT_EQ(test_task_mpi.ValidationImpl(), false);
  }
}

TEST(opolin_d_simple_iteration_method_mpi, test_negative_values) {
  boost::mpi::communicator world;
  int size = 3;
  double epsilon = 1e-8;
  int maxIters = 10000;

  std::vector<double> x_ref, A, b;

  std::vector<double> x_out(size, 0.0);
  auto task_data_mpi = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    A = {5.0, -1.0, 2.0, -1.0, 6.0, -1.0, 2.0, -1.0, 7.0};
    b = {-9.0, -8.0, -21.0};
    x_ref = {-1.0, -2.0, -3.0};
    task_data_mpi->inputs.emplace_back(reinterpret_cast<uint8_t *>(A.data()));
    task_data_mpi->inputs_count.emplace_back(x_out.size());
    task_data_mpi->inputs.emplace_back(reinterpret_cast<uint8_t *>(b.data()));
    task_data_mpi->inputs.emplace_back(reinterpret_cast<uint8_t *>(&epsilon));
    task_data_mpi->inputs.emplace_back(reinterpret_cast<uint8_t *>(&maxIters));
    task_data_mpi->outputs.emplace_back(reinterpret_cast<uint8_t *>(x_out.data()));
    task_data_mpi->outputs_count.emplace_back(x_out.size());
  }
  opolin_d_simple_iteration_method_mpi::TestTaskMPI test_task_mpi(task_data_mpi);

  ASSERT_EQ(test_task_mpi.ValidationImpl(), true);
  test_task_mpi.PreProcessingImpl();
  test_task_mpi.RunImpl();
  test_task_mpi.PostProcessingImpl();
  if (world.rank() == 0) {
    for (size_t i = 0; i < x_ref.size(); ++i) {
      ASSERT_NEAR(x_ref[i], x_out[i], 1e-3);
    }
  }
}

TEST(opolin_d_simple_iteration_method_mpi, test_singular_matrix) {
  int size = 3;
  double epsilon = 1e-8;
  int maxIters = 10000;

  std::vector<double> A, b;

  std::vector<double> x_out(size, 0.0);
  auto task_data_mpi = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    std::vector<double> A = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 5.0, 7.0, 9.0};
    std::vector<double> b = {1.0, 2.0, 3.0};

    task_data_mpi->inputs.emplace_back(reinterpret_cast<uint8_t *>(A.data()));
    task_data_mpi->inputs_count.emplace_back(x_out.size());
    task_data_mpi->inputs.emplace_back(reinterpret_cast<uint8_t *>(b.data()));
    task_data_mpi->inputs.emplace_back(reinterpret_cast<uint8_t *>(&epsilon));
    task_data_mpi->inputs.emplace_back(reinterpret_cast<uint8_t *>(&maxIters));
    task_data_mpi->outputs.emplace_back(reinterpret_cast<uint8_t *>(x_out.data()));
    task_data_mpi->outputs_count.emplace_back(x_out.size());
  }
  opolin_d_simple_iteration_method_mpi::TestTaskMPI test_task_mpi(task_data_mpi);
  if (world.rank() == 0) {
    ASSERT_EQ(test_task_mpi.ValidationImpl(), false);
  }
}

TEST(opolin_d_simple_iteration_method_mpi, test_simple_matrix) {
  int size = 3;
  double epsilon = 1e-8;
  int maxIters = 10000;

  std::vector<double> x_ref, A, b;
  std::vector<double> x_out(size, 0.0);

  auto task_data_mpi = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    A = {1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0};
    b = {1.0, 1.0, 1.0};
    x_ref = {1.0, 1.0, 1.0};

    task_data_mpi->inputs.emplace_back(reinterpret_cast<uint8_t *>(A.data()));
    task_data_mpi->inputs_count.emplace_back(x_out.size());
    task_data_mpi->inputs.emplace_back(reinterpret_cast<uint8_t *>(b.data()));
    task_data_mpi->inputs.emplace_back(reinterpret_cast<uint8_t *>(&epsilon));
    task_data_mpi->inputs.emplace_back(reinterpret_cast<uint8_t *>(&maxIters));
    task_data_mpi->outputs.emplace_back(reinterpret_cast<uint8_t *>(x_out.data()));
    task_data_mpi->outputs_count.emplace_back(x_out.size());
  }
  opolin_d_simple_iteration_method_mpi::TestTaskMPI test_task_mpi(task_data_mpi);

  ASSERT_EQ(test_task_mpi.ValidationImpl(), true);
  test_task_mpi.PreProcessingImpl();
  test_task_mpi.RunImpl();
  test_task_mpi.PostProcessingImpl();
  if (world.rank() == 0) {
    for (size_t i = 0; i < x_ref.size(); ++i) {
      ASSERT_NEAR(x_ref[i], x_out[i], 1e-3);
    }
  }
}

TEST(opolin_d_simple_iteration_method_mpi, test_single_element) {
  int size = 1;
  double epsilon = 1e-8;
  int maxIters = 10000;

  std::vector<double> x_ref, A, b;
  std::vector<double> x_out(size, 0.0);
  auto task_data_mpi = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    A = {1.0};
    b = {10.0};
    x_ref = {10.0};

    task_data_mpi->inputs.emplace_back(reinterpret_cast<uint8_t *>(A.data()));
    task_data_mpi->inputs_count.emplace_back(x_out.size());
    task_data_mpi->inputs.emplace_back(reinterpret_cast<uint8_t *>(b.data()));
    task_data_mpi->inputs.emplace_back(reinterpret_cast<uint8_t *>(&epsilon));
    task_data_mpi->inputs.emplace_back(reinterpret_cast<uint8_t *>(&maxIters));
    task_data_mpi->outputs.emplace_back(reinterpret_cast<uint8_t *>(x_out.data()));
    task_data_mpi->outputs_count.emplace_back(x_out.size());
  }
  opolin_d_simple_iteration_method_mpi::TestTaskMPI test_task_mpi(task_data_mpi);

  ASSERT_EQ(test_task_mpi.ValidationImpl(), true);
  test_task_mpi.PreProcessingImpl();
  test_task_mpi.RunImpl();
  test_task_mpi.PostProcessingImpl();
  if (world.rank() == 0) {
    for (size_t i = 0; i < x_ref.size(); ++i) {
      ASSERT_NEAR(x_ref[i], x_out[i], 1e-3);
    }
  }
}