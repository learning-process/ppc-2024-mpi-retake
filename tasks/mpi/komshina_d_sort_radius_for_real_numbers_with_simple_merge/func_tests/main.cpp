#include <gtest/gtest.h>

#include <algorithm>
#include <boost/mpi/communicator.hpp>
#include <boost/mpi/environment.hpp>
#include <cstdint>
#include <memory>
#include <random>
#include <vector>

#include "core/task/include/task.hpp"
#include "mpi/komshina_d_sort_radius_for_real_numbers_with_simple_merge/include/ops_mpi.hpp"

namespace mpi = boost::mpi;
using namespace komshina_d_sort_radius_for_real_numbers_with_simple_merge_mpi;

TEST(komshina_d_sort_radius_for_real_numbers_with_simple_merge_mpi, EmptyData) {
  mpi::environment env;
  mpi::communicator world;

  std::shared_ptr<ppc::core::TaskData> task_data_mpi = std::make_shared<ppc::core::TaskData>();

  int size = 0;
  std::vector<double> in;
  std::vector<double> out(size, 0.0);

  if (world.rank() == 0) {
    task_data_mpi->inputs = {reinterpret_cast<uint8_t*>(&size), reinterpret_cast<uint8_t*>(in.data())};
    task_data_mpi->inputs_count = {1, static_cast<unsigned int>(size)};
    task_data_mpi->outputs = {reinterpret_cast<uint8_t*>(out.data())};
    task_data_mpi->outputs_count = {static_cast<unsigned int>(size)};
  }

  TestTaskMPI test_task_mpi(task_data_mpi);
  ASSERT_TRUE(test_task_mpi.ValidationImpl());
  test_task_mpi.PreProcessingImpl();
  test_task_mpi.RunImpl();
  test_task_mpi.PostProcessingImpl();

  if (world.rank() == 0) {
    auto* result = reinterpret_cast<double*>(task_data_mpi->outputs[0]);
    ASSERT_EQ(result, out.data());
  }
}

TEST(komshina_d_sort_radius_for_real_numbers_with_simple_merge_mpi, SingleElementData) {
  mpi::environment env;
  mpi::communicator world;

  std::shared_ptr<ppc::core::TaskData> task_data_mpi = std::make_shared<ppc::core::TaskData>();

  int size = 1;
  std::vector<double> in = {42.0};
  std::vector<double> out(size, 0.0);

  if (world.rank() == 0) {
    task_data_mpi->inputs = {reinterpret_cast<uint8_t*>(&size), reinterpret_cast<uint8_t*>(in.data())};
    task_data_mpi->inputs_count = {1, static_cast<unsigned int>(size)};
    task_data_mpi->outputs = {reinterpret_cast<uint8_t*>(out.data())};
    task_data_mpi->outputs_count = {static_cast<unsigned int>(size)};
  }

  TestTaskMPI test_task_mpi(task_data_mpi);
  ASSERT_TRUE(test_task_mpi.ValidationImpl());
  test_task_mpi.PreProcessingImpl();
  test_task_mpi.RunImpl();
  test_task_mpi.PostProcessingImpl();

  if (world.rank() == 0) {
    auto* result = reinterpret_cast<double*>(task_data_mpi->outputs[0]);
    ASSERT_EQ(result[0], in[0]);
  }
}

TEST(komshina_d_sort_radius_for_real_numbers_with_simple_merge_mpi, IncorrectDataSize) {
  mpi::environment env;
  mpi::communicator world;

  std::shared_ptr<ppc::core::TaskData> task_data_mpi = std::make_shared<ppc::core::TaskData>();

  int size = 10;
  std::vector<double> in = {1.0, 2.0, 3.0};
  std::vector<double> out(size, 0.0);

  if (world.rank() == 0) {
    task_data_mpi->inputs = {reinterpret_cast<uint8_t*>(&size), reinterpret_cast<uint8_t*>(in.data())};
    task_data_mpi->inputs_count = {1, static_cast<unsigned int>(in.size())};
    task_data_mpi->outputs = {reinterpret_cast<uint8_t*>(out.data())};
    task_data_mpi->outputs_count = {static_cast<unsigned int>(size)};
  }

  TestTaskMPI test_task_mpi(task_data_mpi);
  ASSERT_FALSE(test_task_mpi.ValidationImpl());
}

TEST(komshina_d_sort_radius_for_real_numbers_with_simple_merge_mpi, LargeData) {
  mpi::environment env;
  mpi::communicator world;

  std::shared_ptr<ppc::core::TaskData> task_data_mpi = std::make_shared<ppc::core::TaskData>();

  int size = 1000;
  std::vector<double> in(size);
  std::vector<double> out(size, 0.0);
  std::mt19937 gen(42);
  std::uniform_real_distribution<double> dist(-1000.0, 1000.0);

  if (world.rank() == 0) {
    for (double& val : in) {
      val = dist(gen);
    }

    task_data_mpi->inputs = {reinterpret_cast<uint8_t*>(&size), reinterpret_cast<uint8_t*>(in.data())};
    task_data_mpi->inputs_count = {1, static_cast<unsigned int>(size)};
    task_data_mpi->outputs = {reinterpret_cast<uint8_t*>(out.data())};
    task_data_mpi->outputs_count = {static_cast<unsigned int>(size)};
  }

  TestTaskMPI test_task_mpi(task_data_mpi);
  ASSERT_TRUE(test_task_mpi.ValidationImpl());
  test_task_mpi.PreProcessingImpl();
  test_task_mpi.RunImpl();
  test_task_mpi.PostProcessingImpl();

  if (world.rank() == 0) {
    auto* result = reinterpret_cast<double*>(task_data_mpi->outputs[0]);
    std::vector<double> expected = in;
    std::ranges::sort(expected);
    for (int i = 0; i < size; ++i) {
      ASSERT_NEAR(result[i], expected[i], 1e-12);
    }
  }
}

TEST(komshina_d_sort_radius_for_real_numbers_with_simple_merge_mpi, AlreadySortedDataWithOddSize) {
  mpi::environment env;
  mpi::communicator world;

  std::shared_ptr<ppc::core::TaskData> task_data_mpi = std::make_shared<ppc::core::TaskData>();

  int size = 11;
  std::vector<double> in = {-5.0, -3.1, -1.2, 0.0, 1.0, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7};
  std::vector<double> out(size, 0.0);

  if (world.rank() == 0) {
    task_data_mpi->inputs = {reinterpret_cast<uint8_t*>(&size), reinterpret_cast<uint8_t*>(in.data())};
    task_data_mpi->inputs_count = {1, static_cast<unsigned int>(size)};
    task_data_mpi->outputs = {reinterpret_cast<uint8_t*>(out.data())};
    task_data_mpi->outputs_count = {static_cast<unsigned int>(size)};
  }

  TestTaskMPI test_task_mpi(task_data_mpi);
  ASSERT_TRUE(test_task_mpi.ValidationImpl());
  test_task_mpi.PreProcessingImpl();
  test_task_mpi.RunImpl();
  test_task_mpi.PostProcessingImpl();

  if (world.rank() == 0) {
    auto* result = reinterpret_cast<double*>(task_data_mpi->outputs[0]);
    for (int i = 0; i < size; ++i) {
      ASSERT_NEAR(result[i], in[i], 1e-12);
    }
  }
}

TEST(komshina_d_sort_radius_for_real_numbers_with_simple_merge_mpi, DataWithNegativeNumbers) {
  mpi::environment env;
  mpi::communicator world;

  std::shared_ptr<ppc::core::TaskData> task_data_mpi = std::make_shared<ppc::core::TaskData>();

  int size = 5;
  std::vector<double> in = {-2.3, 1.5, -0.8, 3.1, -1.2};
  std::vector<double> out(size, 0.0);

  if (world.rank() == 0) {
    task_data_mpi->inputs = {reinterpret_cast<uint8_t*>(&size), reinterpret_cast<uint8_t*>(in.data())};
    task_data_mpi->inputs_count = {1, static_cast<unsigned int>(size)};
    task_data_mpi->outputs = {reinterpret_cast<uint8_t*>(out.data())};
    task_data_mpi->outputs_count = {static_cast<unsigned int>(size)};
  }

  TestTaskMPI test_task_mpi(task_data_mpi);
  ASSERT_TRUE(test_task_mpi.ValidationImpl());
  test_task_mpi.PreProcessingImpl();
  test_task_mpi.RunImpl();
  test_task_mpi.PostProcessingImpl();

  if (world.rank() == 0) {
    auto* result = reinterpret_cast<double*>(task_data_mpi->outputs[0]);
    std::vector<double> expected = in;
    std::ranges::sort(expected);
    for (int i = 0; i < size; ++i) {
      ASSERT_NEAR(result[i], expected[i], 1e-12);
    }
  }
}

TEST(komshina_d_sort_radius_for_real_numbers_with_simple_merge_mpi, SimpleData) {
  mpi::environment env;
  mpi::communicator world;

  std::shared_ptr<ppc::core::TaskData> task_data_mpi = std::make_shared<ppc::core::TaskData>();

  int size = 8;
  std::vector<double> in = {8.3, -4.7, 2.1, 3.5, 0.9, -1.2, 4.4, -5.6};
  std::vector<double> out(size, 0.0);

  if (world.rank() == 0) {
    task_data_mpi->inputs = {reinterpret_cast<uint8_t*>(&size), reinterpret_cast<uint8_t*>(in.data())};
    task_data_mpi->inputs_count = {1, static_cast<unsigned int>(size)};
    task_data_mpi->outputs = {reinterpret_cast<uint8_t*>(out.data())};
    task_data_mpi->outputs_count = {static_cast<unsigned int>(size)};
  }

  TestTaskMPI test_task_mpi(task_data_mpi);
  ASSERT_TRUE(test_task_mpi.ValidationImpl());
  test_task_mpi.PreProcessingImpl();
  test_task_mpi.RunImpl();
  test_task_mpi.PostProcessingImpl();

  if (world.rank() == 0) {
    auto* result = reinterpret_cast<double*>(task_data_mpi->outputs[0]);
    std::vector<double> expected = in;
    std::ranges::sort(expected);
    for (int i = 0; i < size; ++i) {
      ASSERT_NEAR(result[i], expected[i], 1e-12);
    }
  }
}

TEST(komshina_d_sort_radius_for_real_numbers_with_simple_merge_mpi, RandomData) {
  mpi::environment env;
  mpi::communicator world;

  std::shared_ptr<ppc::core::TaskData> task_data_mpi = std::make_shared<ppc::core::TaskData>();

  int size = 100;
  std::vector<double> in(size);
  std::vector<double> out(size, 0.0);
  std::mt19937 gen(42);
  std::uniform_real_distribution<double> dist(-1000.0, 1000.0);

  if (world.rank() == 0) {
    for (double& val : in) {
      val = dist(gen);
    }

    task_data_mpi->inputs = {reinterpret_cast<uint8_t*>(&size), reinterpret_cast<uint8_t*>(in.data())};
    task_data_mpi->inputs_count = {1, static_cast<unsigned int>(size)};
    task_data_mpi->outputs = {reinterpret_cast<uint8_t*>(out.data())};
    task_data_mpi->outputs_count = {static_cast<unsigned int>(size)};
  }

  TestTaskMPI test_task_mpi(task_data_mpi);
  ASSERT_TRUE(test_task_mpi.ValidationImpl());
  test_task_mpi.PreProcessingImpl();
  test_task_mpi.RunImpl();
  test_task_mpi.PostProcessingImpl();

  if (world.rank() == 0) {
    auto* result = reinterpret_cast<double*>(task_data_mpi->outputs[0]);
    std::vector<double> expected = in;
    std::ranges::sort(expected);
    for (int i = 0; i < size; ++i) {
      ASSERT_NEAR(result[i], expected[i], 1e-12);
    }
  }
}

TEST(komshina_d_sort_radius_for_real_numbers_with_simple_merge_mpi, AlreadySortedData) {
  mpi::environment env;
  mpi::communicator world;

  std::shared_ptr<ppc::core::TaskData> task_data_mpi = std::make_shared<ppc::core::TaskData>();

  int size = 10;
  std::vector<double> in = {-5.0, -3.1, -1.2, 0.0, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7};
  std::vector<double> out(size, 0.0);

  if (world.rank() == 0) {
    task_data_mpi->inputs = {reinterpret_cast<uint8_t*>(&size), reinterpret_cast<uint8_t*>(in.data())};
    task_data_mpi->inputs_count = {1, static_cast<unsigned int>(size)};
    task_data_mpi->outputs = {reinterpret_cast<uint8_t*>(out.data())};
    task_data_mpi->outputs_count = {static_cast<unsigned int>(size)};
  }

  TestTaskMPI test_task_mpi(task_data_mpi);
  ASSERT_TRUE(test_task_mpi.ValidationImpl());
  test_task_mpi.PreProcessingImpl();
  test_task_mpi.RunImpl();
  test_task_mpi.PostProcessingImpl();

  if (world.rank() == 0) {
    auto* result = reinterpret_cast<double*>(task_data_mpi->outputs[0]);
    for (int i = 0; i < size; ++i) {
      ASSERT_NEAR(result[i], in[i], 1e-12);
    }
  }
}
