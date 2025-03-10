// Copyright 2023 Nesterov Alexander
#include <gtest/gtest.h>

#include <boost/mpi/collectives.hpp>
#include <boost/mpi/communicator.hpp>
#include <climits>
#include <cstdint>
#include <cstdlib>
#include <ctime>
#include <memory>
#include <random>
#include <vector>

#include "core/task/include/task.hpp"
#include "mpi/malyshev_v_conjugate_gradient/include/ops_mpi.hpp"

namespace malyshev_v_conjugate_gradient_mpi {
namespace {
void GenerateData(size_t size, std::vector<double>& matrix_a, std::vector<double>& vector_b,
                  std::vector<double>& expected) {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::normal_distribution<> dist(-5.0, 5.0);
  std::vector<double> m(size * size);
  for (size_t i = 0; i < size; i++) {
    for (size_t j = 0; j < size; j++) {
      m[(i * size) + j] = dist(gen);
    }
  }
  matrix_a.assign(size * size, 0.0);
  for (size_t i = 0; i < size; i++) {
    for (size_t j = 0; j < size; j++) {
      for (size_t k = 0; k < size; k++) {
        matrix_a[(i * size) + j] += m[(k * size) + i] * m[(k * size) + j];
      }
    }
  }
  for (size_t i = 0; i < size; i++) {
    matrix_a[(i * size) + i] += static_cast<double>(size);
  }
  expected.resize(size);
  for (size_t i = 0; i < size; i++) {
    expected[i] = dist(gen);
  }
  vector_b.assign(size, 0.0);
  for (size_t i = 0; i < size; i++) {
    for (size_t j = 0; j < size; j++) {
      vector_b[i] += matrix_a[(i * size) + j] * expected[j];
    }
  }
}
}  // namespace
}  // namespace malyshev_v_conjugate_gradient_mpi

TEST(malyshev_v_conjugate_gradient_mpi, test_small_system) {
  boost::mpi::communicator world;
  int size = 5;
  double epsilon = 1e-8;

  std::vector<double> x_ref;
  std::vector<double> matrix_a;
  std::vector<double> vector_b;

  std::vector<double> x_out(size, 0.0);
  std::shared_ptr<ppc::core::TaskData> task_data_mpi = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    malyshev_v_conjugate_gradient_mpi::GenerateData(size, matrix_a, vector_b, x_ref);
    task_data_mpi->inputs.emplace_back(reinterpret_cast<uint8_t*>(matrix_a.data()));
    task_data_mpi->inputs_count.emplace_back(x_out.size());
    task_data_mpi->inputs.emplace_back(reinterpret_cast<uint8_t*>(vector_b.data()));
    task_data_mpi->inputs.emplace_back(reinterpret_cast<uint8_t*>(&epsilon));
    task_data_mpi->outputs.emplace_back(reinterpret_cast<uint8_t*>(x_out.data()));
    task_data_mpi->outputs_count.emplace_back(x_out.size());
  }
  malyshev_v_conjugate_gradient_mpi::ConjugateGradientMethod test_task_parallel(task_data_mpi);

  ASSERT_EQ(test_task_parallel.Validation(), true);
  test_task_parallel.PreProcessing();
  test_task_parallel.Run();
  test_task_parallel.PostProcessing();
  if (world.rank() == 0) {
    for (size_t i = 0; i < x_ref.size(); ++i) {
      ASSERT_NEAR(x_ref[i], x_out[i], 1e-3);
    }
  }
}

TEST(malyshev_v_conjugate_gradient_mpi, test_big_system) {
  boost::mpi::communicator world;
  int size = 25;
  double epsilon = 1e-8;

  std::vector<double> x_ref;
  std::vector<double> matrix_a;
  std::vector<double> vector_b;

  std::vector<double> x_out(size, 0.0);
  std::shared_ptr<ppc::core::TaskData> task_data_mpi = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    malyshev_v_conjugate_gradient_mpi::GenerateData(size, matrix_a, vector_b, x_ref);
    task_data_mpi->inputs.emplace_back(reinterpret_cast<uint8_t*>(matrix_a.data()));
    task_data_mpi->inputs_count.emplace_back(x_out.size());
    task_data_mpi->inputs.emplace_back(reinterpret_cast<uint8_t*>(vector_b.data()));
    task_data_mpi->inputs.emplace_back(reinterpret_cast<uint8_t*>(&epsilon));
    task_data_mpi->outputs.emplace_back(reinterpret_cast<uint8_t*>(x_out.data()));
    task_data_mpi->outputs_count.emplace_back(x_out.size());
  }
  malyshev_v_conjugate_gradient_mpi::ConjugateGradientMethod test_task_parallel(task_data_mpi);

  ASSERT_EQ(test_task_parallel.Validation(), true);
  test_task_parallel.PreProcessing();
  test_task_parallel.Run();
  test_task_parallel.PostProcessing();
  if (world.rank() == 0) {
    for (size_t i = 0; i < x_ref.size(); ++i) {
      ASSERT_NEAR(x_ref[i], x_out[i], 1e-3);
    }
  }
}

TEST(malyshev_v_conjugate_gradient_mpi, test_correct_input) {
  boost::mpi::communicator world;
  int size = 3;
  double epsilon = 1e-8;

  std::vector<double> x_ref{1.0, 1.0, 1.0};
  std::vector<double> matrix_a{4.0, 1.0, 2.0, 1.0, 5.0, 1.0, 2.0, 1.0, 5.0};
  std::vector<double> vector_b{7.0, 7.0, 8.0};

  std::vector<double> x_out(size, 0.0);
  auto task_data_mpi = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    task_data_mpi->inputs.emplace_back(reinterpret_cast<uint8_t*>(matrix_a.data()));
    task_data_mpi->inputs_count.emplace_back(x_out.size());
    task_data_mpi->inputs.emplace_back(reinterpret_cast<uint8_t*>(vector_b.data()));
    task_data_mpi->inputs.emplace_back(reinterpret_cast<uint8_t*>(&epsilon));
    task_data_mpi->outputs.emplace_back(reinterpret_cast<uint8_t*>(x_out.data()));
    task_data_mpi->outputs_count.emplace_back(x_out.size());
  }
  malyshev_v_conjugate_gradient_mpi::ConjugateGradientMethod test_task_parallel(task_data_mpi);

  ASSERT_EQ(test_task_parallel.Validation(), true);
  test_task_parallel.PreProcessing();
  test_task_parallel.Run();
  test_task_parallel.PostProcessing();
  if (world.rank() == 0) {
    for (size_t i = 0; i < x_ref.size(); ++i) {
      ASSERT_NEAR(x_ref[i], x_out[i], 1e-3);
    }
  }
}

TEST(malyshev_v_conjugate_gradient_mpi, test_no_symmetric_matrix) {
  boost::mpi::communicator world;
  int size = 3;
  double epsilon = 1e-8;
  auto task_data_mpi = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    std::vector<double> matrix_a{29.0, 0.0, 39.0, 29.0, 53.0, 17.0, 39.0, 1.0, 90.0};
    std::vector<double> vector_b{0.0, 0.0, 0.0};

    std::vector<double> x_out(size, 0.0);

    task_data_mpi->inputs.emplace_back(reinterpret_cast<uint8_t*>(matrix_a.data()));
    task_data_mpi->inputs_count.emplace_back(x_out.size());
    task_data_mpi->inputs.emplace_back(reinterpret_cast<uint8_t*>(vector_b.data()));
    task_data_mpi->inputs.emplace_back(reinterpret_cast<uint8_t*>(&epsilon));
    task_data_mpi->outputs.emplace_back(reinterpret_cast<uint8_t*>(x_out.data()));
    task_data_mpi->outputs_count.emplace_back(x_out.size());

    malyshev_v_conjugate_gradient_mpi::ConjugateGradientMethod test_task_parallel(task_data_mpi);
    ASSERT_EQ(test_task_parallel.Validation(), false);
  }
}

TEST(malyshev_v_conjugate_gradient_mpi, test_negative_values) {
  boost::mpi::communicator world;
  int size = 3;
  double epsilon = 1e-8;

  std::vector<double> x_ref{-0.437926, -1.924931, 0.531806};
  std::vector<double> matrix_a{244.913, -64.084, 59.893, -64.084, 84.215, -23.392, 59.893, -23.392, 31.227};
  std::vector<double> vector_b{47.955, -146.484, 35.406};

  std::vector<double> x_out(size, 0.0);
  auto task_data_mpi = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    task_data_mpi->inputs.emplace_back(reinterpret_cast<uint8_t*>(matrix_a.data()));
    task_data_mpi->inputs_count.emplace_back(x_out.size());
    task_data_mpi->inputs.emplace_back(reinterpret_cast<uint8_t*>(vector_b.data()));
    task_data_mpi->inputs.emplace_back(reinterpret_cast<uint8_t*>(&epsilon));
    task_data_mpi->outputs.emplace_back(reinterpret_cast<uint8_t*>(x_out.data()));
    task_data_mpi->outputs_count.emplace_back(x_out.size());
  }
  malyshev_v_conjugate_gradient_mpi::ConjugateGradientMethod test_task_parallel(task_data_mpi);

  ASSERT_EQ(test_task_parallel.Validation(), true);
  test_task_parallel.PreProcessing();
  test_task_parallel.Run();
  test_task_parallel.PostProcessing();
  if (world.rank() == 0) {
    for (size_t i = 0; i < x_ref.size(); ++i) {
      ASSERT_NEAR(x_ref[i], x_out[i], 1e-3);
    }
  }
}

TEST(malyshev_v_conjugate_gradient_mpi, test_no_positive_definite_matrix) {
  boost::mpi::communicator world;
  int size = 3;
  double epsilon = 1e-8;

  std::vector<double> matrix_a{0.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 0.0};
  std::vector<double> vector_b{0.0, 0.0, 0.0};

  std::vector<double> x_out(size, 0.0);
  auto task_data_mpi = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    task_data_mpi->inputs.emplace_back(reinterpret_cast<uint8_t*>(matrix_a.data()));
    task_data_mpi->inputs_count.emplace_back(x_out.size());
    task_data_mpi->inputs.emplace_back(reinterpret_cast<uint8_t*>(vector_b.data()));
    task_data_mpi->inputs.emplace_back(reinterpret_cast<uint8_t*>(&epsilon));
    task_data_mpi->outputs.emplace_back(reinterpret_cast<uint8_t*>(x_out.data()));
    task_data_mpi->outputs_count.emplace_back(x_out.size());

    malyshev_v_conjugate_gradient_mpi::ConjugateGradientMethod test_task_parallel(task_data_mpi);
    ASSERT_EQ(test_task_parallel.Validation(), false);
  }
}

TEST(malyshev_v_conjugate_gradient_mpi, test_simple_matrix) {
  boost::mpi::communicator world;
  int size = 3;
  double epsilon = 1e-8;

  std::vector<double> x_ref{1.0, 1.0, 1.0};
  std::vector<double> matrix_a{1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0};
  std::vector<double> vector_b{1.0, 1.0, 1.0};

  std::vector<double> x_out(size, 0.0);
  auto task_data_mpi = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    task_data_mpi->inputs.emplace_back(reinterpret_cast<uint8_t*>(matrix_a.data()));
    task_data_mpi->inputs_count.emplace_back(x_out.size());
    task_data_mpi->inputs.emplace_back(reinterpret_cast<uint8_t*>(vector_b.data()));
    task_data_mpi->inputs.emplace_back(reinterpret_cast<uint8_t*>(&epsilon));
    task_data_mpi->outputs.emplace_back(reinterpret_cast<uint8_t*>(x_out.data()));
    task_data_mpi->outputs_count.emplace_back(x_out.size());
  }
  malyshev_v_conjugate_gradient_mpi::ConjugateGradientMethod test_task_parallel(task_data_mpi);

  ASSERT_EQ(test_task_parallel.Validation(), true);
  test_task_parallel.PreProcessing();
  test_task_parallel.Run();
  test_task_parallel.PostProcessing();
  if (world.rank() == 0) {
    for (size_t i = 0; i < x_ref.size(); ++i) {
      ASSERT_NEAR(x_ref[i], x_out[i], 1e-3);
    }
  }
}

TEST(malyshev_v_conjugate_gradient_mpi, test_single_element) {
  boost::mpi::communicator world;
  int size = 1;
  double epsilon = 1e-5;

  std::vector<double> x_ref{10.0};
  std::vector<double> matrix_a{1.0};
  std::vector<double> vector_b{10.0};

  std::vector<double> x_out(size, 0.0);
  auto task_data_mpi = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    task_data_mpi->inputs.emplace_back(reinterpret_cast<uint8_t*>(matrix_a.data()));
    task_data_mpi->inputs_count.emplace_back(x_out.size());
    task_data_mpi->inputs.emplace_back(reinterpret_cast<uint8_t*>(vector_b.data()));
    task_data_mpi->inputs.emplace_back(reinterpret_cast<uint8_t*>(&epsilon));
    task_data_mpi->outputs.emplace_back(reinterpret_cast<uint8_t*>(x_out.data()));
    task_data_mpi->outputs_count.emplace_back(x_out.size());
  }
  malyshev_v_conjugate_gradient_mpi::ConjugateGradientMethod test_task_parallel(task_data_mpi);

  ASSERT_EQ(test_task_parallel.Validation(), true);
  test_task_parallel.PreProcessing();
  test_task_parallel.Run();
  test_task_parallel.PostProcessing();
  if (world.rank() == 0) {
    for (size_t i = 0; i < x_ref.size(); ++i) {
      ASSERT_NEAR(x_ref[i], x_out[i], 1e-3);
    }
  }
}