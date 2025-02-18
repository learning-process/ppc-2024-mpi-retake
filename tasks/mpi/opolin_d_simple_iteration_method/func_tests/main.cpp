// Copyright 2023 Nesterov Alexander
#include <gtest/gtest.h>

#include <boost/mpi/collectives.hpp>
#include <boost/mpi/communicator.hpp>
#include <climits>
#include <cstdint>
#include <cstdlib>
#include <ctime>
#include <memory>
#include <vector>

#include "core/task/include/task.hpp"
#include "mpi/opolin_d_simple_iteration_method/include/ops_mpi.hpp"

namespace opolin_d_simple_iteration_method_mpi {
namespace {
void GenerateTestData(size_t size, std::vector<double> &x, std::vector<double> &a, std::vector<double> &b) {
  std::srand(static_cast<unsigned>(std::time(nullptr)));

  x.resize(size);
  for (size_t i = 0; i < size; ++i) {
    x[i] = -10.0 + static_cast<double>(std::rand() % 1000) / 50.0;
  }

  a.resize(size * size, 0.0);
  for (size_t i = 0; i < size; ++i) {
    double sum = 0.0;
    for (size_t j = 0; j < size; ++j) {
      if (i != j) {
        a[(i * size) + j] = -1.0 + static_cast<double>(std::rand() % 1000) / 500.0;
        sum += std::abs(a[(i * size) + j]);
      }
    }
    a[(i * size) + i] = sum + 1.0;
  }
  b.resize(size, 0.0);
  for (size_t i = 0; i < size; ++i) {
    for (size_t j = 0; j < size; ++j) {
      b[i] += a[(i * size) + j] * x[j];
    }
  }
}
}  // namespace
}  // namespace opolin_d_simple_iteration_method_mpi

TEST(opolin_d_simple_iteration_method_mpi, test_small_system) {
  boost::mpi::communicator world;
  int size = 5;
  double epsilon = 1e-8;
  int max_iters = 10000;

  std::vector<double> x_ref;
  std::vector<double> a;
  std::vector<double> b;

  std::vector<double> x_out(size, 0.0);
  std::shared_ptr<ppc::core::TaskData> task_data_mpi = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    opolin_d_simple_iteration_method_mpi::GenerateTestData(size, x_ref, a, b);
    task_data_mpi->inputs.emplace_back(reinterpret_cast<uint8_t *>(a.data()));
    task_data_mpi->inputs_count.emplace_back(x_out.size());
    task_data_mpi->inputs.emplace_back(reinterpret_cast<uint8_t *>(b.data()));
    task_data_mpi->inputs.emplace_back(reinterpret_cast<uint8_t *>(&epsilon));
    task_data_mpi->inputs.emplace_back(reinterpret_cast<uint8_t *>(&max_iters));
    task_data_mpi->outputs.emplace_back(reinterpret_cast<uint8_t *>(x_out.data()));
    task_data_mpi->outputs_count.emplace_back(x_out.size());
  }
  opolin_d_simple_iteration_method_mpi::SimpleIterMethodkMPI test_task_parallel(task_data_mpi);

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

TEST(opolin_d_simple_iteration_method_mpi, test_big_system) {
  boost::mpi::communicator world;
  int size = 50;
  double epsilon = 1e-8;
  int max_iters = 10000;

  std::vector<double> x_ref;
  std::vector<double> a;
  std::vector<double> b;

  std::vector<double> x_out(size, 0.0);
  auto task_data_mpi = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    opolin_d_simple_iteration_method_mpi::GenerateTestData(size, x_ref, a, b);
    task_data_mpi->inputs.emplace_back(reinterpret_cast<uint8_t *>(a.data()));
    task_data_mpi->inputs_count.emplace_back(x_out.size());
    task_data_mpi->inputs.emplace_back(reinterpret_cast<uint8_t *>(b.data()));
    task_data_mpi->inputs.emplace_back(reinterpret_cast<uint8_t *>(&epsilon));
    task_data_mpi->inputs.emplace_back(reinterpret_cast<uint8_t *>(&max_iters));
    task_data_mpi->outputs.emplace_back(reinterpret_cast<uint8_t *>(x_out.data()));
    task_data_mpi->outputs_count.emplace_back(x_out.size());
  }
  opolin_d_simple_iteration_method_mpi::SimpleIterMethodkMPI test_task_mpi(task_data_mpi);

  ASSERT_EQ(test_task_mpi.Validation(), true);
  test_task_mpi.PreProcessing();
  test_task_mpi.Run();
  test_task_mpi.PostProcessing();
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
  int max_iters = 10000;

  std::vector<double> x_ref;
  std::vector<double> a;
  std::vector<double> b;
  std::vector<double> x_out(size, 0.0);
  auto task_data_mpi = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    a = {4.0, 1.0, 2.0, 1.0, 5.0, 1.0, 2.0, 1.0, 5.0};
    b = {7.0, 7.0, 8.0};
    x_ref = {1.0, 1.0, 1.0};

    task_data_mpi->inputs.emplace_back(reinterpret_cast<uint8_t *>(a.data()));
    task_data_mpi->inputs_count.emplace_back(x_out.size());
    task_data_mpi->inputs.emplace_back(reinterpret_cast<uint8_t *>(b.data()));
    task_data_mpi->inputs.emplace_back(reinterpret_cast<uint8_t *>(&epsilon));
    task_data_mpi->inputs.emplace_back(reinterpret_cast<uint8_t *>(&max_iters));
    task_data_mpi->outputs.emplace_back(reinterpret_cast<uint8_t *>(x_out.data()));
    task_data_mpi->outputs_count.emplace_back(x_out.size());
  }
  opolin_d_simple_iteration_method_mpi::SimpleIterMethodkMPI test_task_mpi(task_data_mpi);

  ASSERT_EQ(test_task_mpi.Validation(), true);
  test_task_mpi.PreProcessing();
  test_task_mpi.Run();
  test_task_mpi.PostProcessing();
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
  int max_iters = 1000;
  auto task_data_mpi = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    std::vector<double> a = {3.0, 2.0, 4.0, 1.0, 2.0, 4.0, 1.0, 2.0, 3.0};
    std::vector<double> b = {3.0, 2.0, 2.0};

    std::vector<double> x_out(size, 0.0);

    task_data_mpi->inputs.emplace_back(reinterpret_cast<uint8_t *>(a.data()));
    task_data_mpi->inputs_count.emplace_back(x_out.size());
    task_data_mpi->inputs.emplace_back(reinterpret_cast<uint8_t *>(b.data()));
    task_data_mpi->inputs.emplace_back(reinterpret_cast<uint8_t *>(&epsilon));
    task_data_mpi->inputs.emplace_back(reinterpret_cast<uint8_t *>(&max_iters));
    task_data_mpi->outputs.emplace_back(reinterpret_cast<uint8_t *>(x_out.data()));
    task_data_mpi->outputs_count.emplace_back(x_out.size());

    opolin_d_simple_iteration_method_mpi::SimpleIterMethodkMPI test_task_mpi(task_data_mpi);

    ASSERT_EQ(test_task_mpi.Validation(), false);
  }
}

TEST(opolin_d_simple_iteration_method_mpi, test_negative_values) {
  boost::mpi::communicator world;
  int size = 3;
  double epsilon = 1e-8;
  int max_iters = 10000;

  std::vector<double> x_ref;
  std::vector<double> a;
  std::vector<double> b;

  std::vector<double> x_out(size, 0.0);
  auto task_data_mpi = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    a = {5.0, -1.0, 2.0, -1.0, 6.0, -1.0, 2.0, -1.0, 7.0};
    b = {-9.0, -8.0, -21.0};
    x_ref = {-1.0, -2.0, -3.0};
    task_data_mpi->inputs.emplace_back(reinterpret_cast<uint8_t *>(a.data()));
    task_data_mpi->inputs_count.emplace_back(x_out.size());
    task_data_mpi->inputs.emplace_back(reinterpret_cast<uint8_t *>(b.data()));
    task_data_mpi->inputs.emplace_back(reinterpret_cast<uint8_t *>(&epsilon));
    task_data_mpi->inputs.emplace_back(reinterpret_cast<uint8_t *>(&max_iters));
    task_data_mpi->outputs.emplace_back(reinterpret_cast<uint8_t *>(x_out.data()));
    task_data_mpi->outputs_count.emplace_back(x_out.size());
  }
  opolin_d_simple_iteration_method_mpi::SimpleIterMethodkMPI test_task_mpi(task_data_mpi);

  ASSERT_EQ(test_task_mpi.Validation(), true);
  test_task_mpi.PreProcessing();
  test_task_mpi.Run();
  test_task_mpi.PostProcessing();
  if (world.rank() == 0) {
    for (size_t i = 0; i < x_ref.size(); ++i) {
      ASSERT_NEAR(x_ref[i], x_out[i], 1e-3);
    }
  }
}

TEST(opolin_d_simple_iteration_method_mpi, test_singular_matrix) {
  boost::mpi::communicator world;
  int size = 3;
  double epsilon = 1e-8;
  int max_iters = 10000;

  std::vector<double> a;
  std::vector<double> b;

  std::vector<double> x_out(size, 0.0);
  auto task_data_mpi = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    a = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 5.0, 7.0, 9.0};
    b = {1.0, 2.0, 3.0};

    task_data_mpi->inputs.emplace_back(reinterpret_cast<uint8_t *>(a.data()));
    task_data_mpi->inputs_count.emplace_back(x_out.size());
    task_data_mpi->inputs.emplace_back(reinterpret_cast<uint8_t *>(b.data()));
    task_data_mpi->inputs.emplace_back(reinterpret_cast<uint8_t *>(&epsilon));
    task_data_mpi->inputs.emplace_back(reinterpret_cast<uint8_t *>(&max_iters));
    task_data_mpi->outputs.emplace_back(reinterpret_cast<uint8_t *>(x_out.data()));
    task_data_mpi->outputs_count.emplace_back(x_out.size());
  }
  opolin_d_simple_iteration_method_mpi::SimpleIterMethodkMPI test_task_mpi(task_data_mpi);
  if (world.rank() == 0) {
    ASSERT_EQ(test_task_mpi.Validation(), false);
  }
}

TEST(opolin_d_simple_iteration_method_mpi, test_simple_matrix) {
  boost::mpi::communicator world;
  int size = 3;
  double epsilon = 1e-8;
  int max_iters = 10000;

  std::vector<double> x_ref;
  std::vector<double> a;
  std::vector<double> b;
  std::vector<double> x_out(size, 0.0);

  auto task_data_mpi = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    a = {1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0};
    b = {1.0, 1.0, 1.0};
    x_ref = {1.0, 1.0, 1.0};

    task_data_mpi->inputs.emplace_back(reinterpret_cast<uint8_t *>(a.data()));
    task_data_mpi->inputs_count.emplace_back(x_out.size());
    task_data_mpi->inputs.emplace_back(reinterpret_cast<uint8_t *>(b.data()));
    task_data_mpi->inputs.emplace_back(reinterpret_cast<uint8_t *>(&epsilon));
    task_data_mpi->inputs.emplace_back(reinterpret_cast<uint8_t *>(&max_iters));
    task_data_mpi->outputs.emplace_back(reinterpret_cast<uint8_t *>(x_out.data()));
    task_data_mpi->outputs_count.emplace_back(x_out.size());
  }
  opolin_d_simple_iteration_method_mpi::SimpleIterMethodkMPI test_task_mpi(task_data_mpi);
  ASSERT_EQ(test_task_mpi.Validation(), true);
  test_task_mpi.PreProcessing();
  test_task_mpi.Run();
  test_task_mpi.PostProcessing();
  if (world.rank() == 0) {
    for (size_t i = 0; i < x_ref.size(); ++i) {
      ASSERT_NEAR(x_ref[i], x_out[i], 1e-3);
    }
  }
}

TEST(opolin_d_simple_iteration_method_mpi, test_single_element) {
  boost::mpi::communicator world;
  int size = 1;
  double epsilon = 1e-8;
  int max_iters = 10000;

  std::vector<double> x_ref;
  std::vector<double> a;
  std::vector<double> b;
  std::vector<double> x_out(size, 0.0);
  auto task_data_mpi = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    a = {1.0};
    b = {10.0};
    x_ref = {10.0};

    task_data_mpi->inputs.emplace_back(reinterpret_cast<uint8_t *>(a.data()));
    task_data_mpi->inputs_count.emplace_back(x_out.size());
    task_data_mpi->inputs.emplace_back(reinterpret_cast<uint8_t *>(b.data()));
    task_data_mpi->inputs.emplace_back(reinterpret_cast<uint8_t *>(&epsilon));
    task_data_mpi->inputs.emplace_back(reinterpret_cast<uint8_t *>(&max_iters));
    task_data_mpi->outputs.emplace_back(reinterpret_cast<uint8_t *>(x_out.data()));
    task_data_mpi->outputs_count.emplace_back(x_out.size());
  }
  opolin_d_simple_iteration_method_mpi::SimpleIterMethodkMPI test_task_mpi(task_data_mpi);

  ASSERT_EQ(test_task_mpi.Validation(), true);
  test_task_mpi.PreProcessing();
  test_task_mpi.Run();
  test_task_mpi.PostProcessing();
  if (world.rank() == 0) {
    for (size_t i = 0; i < x_ref.size(); ++i) {
      ASSERT_NEAR(x_ref[i], x_out[i], 1e-3);
    }
  }
}