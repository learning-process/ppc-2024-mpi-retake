#include <gtest/gtest.h>

#include <boost/mpi/communicator.hpp>
#include <boost/mpi/environment.hpp>
#include <boost/serialization/map.hpp>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <memory>
#include <string>
#include <vector>
#include <random>

#include "core/task/include/task.hpp"
#include "core/util/include/util.hpp"
#include "mpi/kavtorev_d_most_different_neighbor_elements/include/ops_mpi.hpp"

namespace kavtorev_d_most_different_neighbor_elements_mpi {
std::vector<int> generator(int sz) {
  std::random_device dev;
  std::mt19937 gen(dev());

  std::vector<int> ans(sz);
  for (int i = 0; i < sz; ++i) {
    ans[i] = gen() % 1000;
    int x = gen() % 2;
    if (x == 0) ans[i] *= -1;
  }

  return ans;
}
}  // namespace kavtorev_d_most_different_neighbor_elements_mpi
TEST(kavtorev_d_most_different_neighbor_elements_mpi, SingleElementInput_ReturnsFalse) {
  boost::mpi::communicator world;
  std::vector<int> global_vec(1, 42);
  std::vector<int> global_max(1);

  auto task_data_mpi = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    task_data_mpi->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vec.data()));
    task_data_mpi->inputs_count.emplace_back(global_vec.size());
    task_data_mpi->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_max.data()));
    task_data_mpi->outputs_count.emplace_back(global_max.size());
  }

  kavtorev_d_most_different_neighbor_elements_mpi::most_different_neighbor_elements_mpi testMpiTaskParallel(
      task_data_mpi);
  ASSERT_EQ(testMpiTaskParallel.ValidationImpl(), false);
}

TEST(kavtorev_d_most_different_neighbor_elements_mpi, LargePositiveNumbers_ReturnsCorrectPair) {
  boost::mpi::communicator world;
  std::vector<int> global_vec;
  std::vector<int> global_max(1);

  auto task_data_mpi = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    global_vec = {1000, 2000, 3000, 4000, 5000};
    task_data_mpi->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vec.data()));
    task_data_mpi->inputs_count.emplace_back(global_vec.size());
    task_data_mpi->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_max.data()));
    task_data_mpi->outputs_count.emplace_back(global_max.size());
  }

  kavtorev_d_most_different_neighbor_elements_mpi::most_different_neighbor_elements_mpi testMpiTaskParallel(
      task_data_mpi);
  ASSERT_EQ(testMpiTaskParallel.ValidationImpl(), true);
  testMpiTaskParallel.PreProcessingImpl();
  testMpiTaskParallel.RunImpl();
  testMpiTaskParallel.PostProcessingImpl();

  if (world.rank() == 0) {
    std::vector<int> reference_max(1);

    auto task_data_seq = std::make_shared<ppc::core::TaskData>();
    task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vec.data()));
    task_data_seq->inputs_count.emplace_back(global_vec.size());
    task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t*>(reference_max.data()));
    task_data_seq->outputs_count.emplace_back(reference_max.size());

    kavtorev_d_most_different_neighbor_elements_mpi::most_different_neighbor_elements_seq test_task_sequential(
        task_data_seq);
    ASSERT_EQ(test_task_sequential.ValidationImpl(), true);
    test_task_sequential.PreProcessingImpl();
    test_task_sequential.RunImpl();
    test_task_sequential.PostProcessingImpl();

    ASSERT_EQ(reference_max[0], global_max[0]);
  }
}

TEST(kavtorev_d_most_different_neighbor_elements_mpi, MixedPositiveAndNegativeNumbers_ReturnsCorrectPair) {
  boost::mpi::communicator world;
  std::vector<int> global_vec;
  std::vector<int> global_max(1);

  auto task_data_mpi = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    global_vec = {-10, 20, -30, 40, -50};
    task_data_mpi->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vec.data()));
    task_data_mpi->inputs_count.emplace_back(global_vec.size());
    task_data_mpi->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_max.data()));
    task_data_mpi->outputs_count.emplace_back(global_max.size());
  }

  kavtorev_d_most_different_neighbor_elements_mpi::most_different_neighbor_elements_mpi testMpiTaskParallel(
      task_data_mpi);
  ASSERT_EQ(testMpiTaskParallel.ValidationImpl(), true);
  testMpiTaskParallel.PreProcessingImpl();
  testMpiTaskParallel.RunImpl();
  testMpiTaskParallel.PostProcessingImpl();

  if (world.rank() == 0) {
    std::vector<int> reference_max(1);

    auto task_data_seq = std::make_shared<ppc::core::TaskData>();
    task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vec.data()));
    task_data_seq->inputs_count.emplace_back(global_vec.size());
    task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t*>(reference_max.data()));
    task_data_seq->outputs_count.emplace_back(reference_max.size());

    kavtorev_d_most_different_neighbor_elements_mpi::most_different_neighbor_elements_seq test_task_sequential(
        task_data_seq);
    ASSERT_EQ(test_task_sequential.ValidationImpl(), true);
    test_task_sequential.PreProcessingImpl();
    test_task_sequential.RunImpl();
    test_task_sequential.PostProcessingImpl();

    ASSERT_EQ(reference_max[0], global_max[0]);
  }
}

TEST(kavtorev_d_most_different_neighbor_elements_mpi, AlternatingPositiveAndNegativeNumbers_ReturnsCorrectPair) {
  boost::mpi::communicator world;
  std::vector<int> global_vec;
  std::vector<int> global_max(1);

  auto task_data_mpi = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    global_vec = {1, -1, 2, -2, 3, -3, 4, -4, 5, -5};
    task_data_mpi->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vec.data()));
    task_data_mpi->inputs_count.emplace_back(global_vec.size());
    task_data_mpi->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_max.data()));
    task_data_mpi->outputs_count.emplace_back(global_max.size());
  }

  kavtorev_d_most_different_neighbor_elements_mpi::most_different_neighbor_elements_mpi testMpiTaskParallel(
      task_data_mpi);
  ASSERT_EQ(testMpiTaskParallel.ValidationImpl(), true);
  testMpiTaskParallel.PreProcessingImpl();
  testMpiTaskParallel.RunImpl();
  testMpiTaskParallel.PostProcessingImpl();

  if (world.rank() == 0) {
    std::vector<int> reference_max(1);

    auto task_data_seq = std::make_shared<ppc::core::TaskData>();
    task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vec.data()));
    task_data_seq->inputs_count.emplace_back(global_vec.size());
    task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t*>(reference_max.data()));
    task_data_seq->outputs_count.emplace_back(reference_max.size());

    kavtorev_d_most_different_neighbor_elements_mpi::most_different_neighbor_elements_seq test_task_sequential(
        task_data_seq);
    ASSERT_EQ(test_task_sequential.ValidationImpl(), true);
    test_task_sequential.PreProcessingImpl();
    test_task_sequential.RunImpl();
    test_task_sequential.PostProcessingImpl();

    ASSERT_EQ(reference_max[0], global_max[0]);
  }
}

TEST(kavtorev_d_most_different_neighbor_elements_mpi, LargeInputSize_ReturnsCorrectPair) {
  boost::mpi::communicator world;
  std::vector<int> global_vec;
  std::vector<int> global_max(1);

  auto task_data_mpi = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    global_vec = kavtorev_d_most_different_neighbor_elements_mpi::generator(1000);
    task_data_mpi->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vec.data()));
    task_data_mpi->inputs_count.emplace_back(global_vec.size());
    task_data_mpi->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_max.data()));
    task_data_mpi->outputs_count.emplace_back(global_max.size());
  }

  kavtorev_d_most_different_neighbor_elements_mpi::most_different_neighbor_elements_mpi testMpiTaskParallel(
      task_data_mpi);
  ASSERT_EQ(testMpiTaskParallel.ValidationImpl(), true);
  testMpiTaskParallel.PreProcessingImpl();
  testMpiTaskParallel.RunImpl();
  testMpiTaskParallel.PostProcessingImpl();

  if (world.rank() == 0) {
    std::vector<int> reference_max(1);

    auto task_data_seq = std::make_shared<ppc::core::TaskData>();
    task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vec.data()));
    task_data_seq->inputs_count.emplace_back(global_vec.size());
    task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t*>(reference_max.data()));
    task_data_seq->outputs_count.emplace_back(reference_max.size());

    kavtorev_d_most_different_neighbor_elements_mpi::most_different_neighbor_elements_seq test_task_sequential(
        task_data_seq);
    ASSERT_EQ(test_task_sequential.ValidationImpl(), true);
    test_task_sequential.PreProcessingImpl();
    test_task_sequential.RunImpl();
    test_task_sequential.PostProcessingImpl();

    ASSERT_EQ(reference_max[0], global_max[0]);
  }
}

TEST(kavtorev_d_most_different_neighbor_elements_mpi, AllNegativeNumbers_ReturnsCorrectPair) {
  boost::mpi::communicator world;
  std::vector<int> global_vec;
  std::vector<int> global_max(1);

  auto task_data_mpi = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    global_vec = {-10, -20, -30, -40, -50};
    task_data_mpi->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vec.data()));
    task_data_mpi->inputs_count.emplace_back(global_vec.size());
    task_data_mpi->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_max.data()));
    task_data_mpi->outputs_count.emplace_back(global_max.size());
  }

  kavtorev_d_most_different_neighbor_elements_mpi::most_different_neighbor_elements_mpi testMpiTaskParallel(
      task_data_mpi);
  ASSERT_EQ(testMpiTaskParallel.ValidationImpl(), true);
  testMpiTaskParallel.PreProcessingImpl();
  testMpiTaskParallel.RunImpl();
  testMpiTaskParallel.PostProcessingImpl();

  if (world.rank() == 0) {
    std::vector<int> reference_max(1);

    auto task_data_seq = std::make_shared<ppc::core::TaskData>();
    task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vec.data()));
    task_data_seq->inputs_count.emplace_back(global_vec.size());
    task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t*>(reference_max.data()));
    task_data_seq->outputs_count.emplace_back(reference_max.size());

    kavtorev_d_most_different_neighbor_elements_mpi::most_different_neighbor_elements_seq test_task_sequential(
        task_data_seq);
    ASSERT_EQ(test_task_sequential.ValidationImpl(), true);
    test_task_sequential.PreProcessingImpl();
    test_task_sequential.RunImpl();
    test_task_sequential.PostProcessingImpl();

    ASSERT_EQ(reference_max[0], global_max[0]);
  }
}

TEST(kavtorev_d_most_different_neighbor_elements_mpi, RandomNumbers_ReturnsCorrectPair) {
  boost::mpi::communicator world;
  std::vector<int> global_vec;
  std::vector<int> global_max(1);

  auto task_data_mpi = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    global_vec = kavtorev_d_most_different_neighbor_elements_mpi::generator(10);
    task_data_mpi->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vec.data()));
    task_data_mpi->inputs_count.emplace_back(global_vec.size());
    task_data_mpi->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_max.data()));
    task_data_mpi->outputs_count.emplace_back(global_max.size());
  }

  kavtorev_d_most_different_neighbor_elements_mpi::most_different_neighbor_elements_mpi testMpiTaskParallel(
      task_data_mpi);
  ASSERT_EQ(testMpiTaskParallel.ValidationImpl(), true);
  testMpiTaskParallel.PreProcessingImpl();
  testMpiTaskParallel.RunImpl();
  testMpiTaskParallel.PostProcessingImpl();

  if (world.rank() == 0) {
    std::vector<int> reference_max(1);

    auto task_data_seq = std::make_shared<ppc::core::TaskData>();
    task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vec.data()));
    task_data_seq->inputs_count.emplace_back(global_vec.size());
    task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t*>(reference_max.data()));
    task_data_seq->outputs_count.emplace_back(reference_max.size());

    kavtorev_d_most_different_neighbor_elements_mpi::most_different_neighbor_elements_seq test_task_sequential(
        task_data_seq);
    ASSERT_EQ(test_task_sequential.ValidationImpl(), true);
    test_task_sequential.PreProcessingImpl();
    test_task_sequential.RunImpl();
    test_task_sequential.PostProcessingImpl();

    ASSERT_EQ(reference_max[0], global_max[0]);
  }
}

TEST(kavtorev_d_most_different_neighbor_elements_mpi, RepeatedNumbers_ReturnsCorrectPair) {
  boost::mpi::communicator world;
  std::vector<int> global_vec;
  std::vector<int> global_max(1);

  auto task_data_mpi = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    global_vec = {5, 5, 5, 5, 5};
    task_data_mpi->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vec.data()));
    task_data_mpi->inputs_count.emplace_back(global_vec.size());
    task_data_mpi->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_max.data()));
    task_data_mpi->outputs_count.emplace_back(global_max.size());
  }

  kavtorev_d_most_different_neighbor_elements_mpi::most_different_neighbor_elements_mpi testMpiTaskParallel(
      task_data_mpi);
  ASSERT_EQ(testMpiTaskParallel.ValidationImpl(), true);
  testMpiTaskParallel.PreProcessingImpl();
  testMpiTaskParallel.RunImpl();
  testMpiTaskParallel.PostProcessingImpl();

  if (world.rank() == 0) {
    std::vector<int> reference_max(1);

    auto task_data_seq = std::make_shared<ppc::core::TaskData>();
    task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vec.data()));
    task_data_seq->inputs_count.emplace_back(global_vec.size());
    task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t*>(reference_max.data()));
    task_data_seq->outputs_count.emplace_back(reference_max.size());

    kavtorev_d_most_different_neighbor_elements_mpi::most_different_neighbor_elements_seq test_task_sequential(
        task_data_seq);
    ASSERT_EQ(test_task_sequential.ValidationImpl(), true);
    test_task_sequential.PreProcessingImpl();
    test_task_sequential.RunImpl();
    test_task_sequential.PostProcessingImpl();

    ASSERT_EQ(reference_max[0], global_max[0]);
  }
}

TEST(kavtorev_d_most_different_neighbor_elements_mpi, LargeRangeNumbers_ReturnsCorrectPair) {
  boost::mpi::communicator world;
  std::vector<int> global_vec;
  std::vector<int> global_max(1);

  auto task_data_mpi = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    global_vec = {-1000000, 1000000};
    task_data_mpi->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vec.data()));
    task_data_mpi->inputs_count.emplace_back(global_vec.size());
    task_data_mpi->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_max.data()));
    task_data_mpi->outputs_count.emplace_back(global_max.size());
  }

  kavtorev_d_most_different_neighbor_elements_mpi::most_different_neighbor_elements_mpi testMpiTaskParallel(
      task_data_mpi);
  ASSERT_EQ(testMpiTaskParallel.ValidationImpl(), true);
  testMpiTaskParallel.PreProcessingImpl();
  testMpiTaskParallel.RunImpl();
  testMpiTaskParallel.PostProcessingImpl();

  if (world.rank() == 0) {
    std::vector<int> reference_max(1);

    auto task_data_seq = std::make_shared<ppc::core::TaskData>();
    task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vec.data()));
    task_data_seq->inputs_count.emplace_back(global_vec.size());
    task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t*>(reference_max.data()));
    task_data_seq->outputs_count.emplace_back(reference_max.size());

    kavtorev_d_most_different_neighbor_elements_mpi::most_different_neighbor_elements_seq test_task_sequential(
        task_data_seq);
    ASSERT_EQ(test_task_sequential.ValidationImpl(), true);
    test_task_sequential.PreProcessingImpl();
    test_task_sequential.RunImpl();
    test_task_sequential.PostProcessingImpl();

    ASSERT_EQ(reference_max[0], global_max[0]);
  }
}

TEST(kavtorev_d_most_different_neighbor_elements_mpi, LargePositiveNumbers_ReturnsCorrectPair) {
  boost::mpi::communicator world;
  std::vector<int> global_vec;
  std::vector<int> global_max(1);

  auto task_data_mpi = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    global_vec = {1000, 2000, 3000, 4000, 5000};
    task_data_mpi->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vec.data()));
    task_data_mpi->inputs_count.emplace_back(global_vec.size());
    task_data_mpi->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_max.data()));
    task_data_mpi->outputs_count.emplace_back(global_max.size());
  }

  kavtorev_d_most_different_neighbor_elements_mpi::most_different_neighbor_elements_mpi testMpiTaskParallel(
      task_data_mpi);
  ASSERT_EQ(testMpiTaskParallel.ValidationImpl(), true);
  testMpiTaskParallel.PreProcessingImpl();
  testMpiTaskParallel.RunImpl();
  testMpiTaskParallel.PostProcessingImpl();

  if (world.rank() == 0) {
    std::vector<int> reference_max(1);

    auto task_data_seq = std::make_shared<ppc::core::TaskData>();
    task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vec.data()));
    task_data_seq->inputs_count.emplace_back(global_vec.size());
    task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t*>(reference_max.data()));
    task_data_seq->outputs_count.emplace_back(reference_max.size());

    kavtorev_d_most_different_neighbor_elements_mpi::most_different_neighbor_elements_seq test_task_sequential(
        task_data_seq);
    ASSERT_EQ(test_task_sequential.ValidationImpl(), true);
    test_task_sequential.PreProcessingImpl();
    test_task_sequential.RunImpl();
    test_task_sequential.PostProcessingImpl();

    ASSERT_EQ(reference_max[0], global_max[0]);
  }
}

TEST(kavtorev_d_most_different_neighbor_elements_mpi, MixedPositiveAndNegativeNumbers_ReturnsCorrectPair) {
  boost::mpi::communicator world;
  std::vector<int> global_vec;
  std::vector<int> global_max(1);

  auto task_data_mpi = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    global_vec = {-10, 20, -30, 40, -50};
    task_data_mpi->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vec.data()));
    task_data_mpi->inputs_count.emplace_back(global_vec.size());
    task_data_mpi->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_max.data()));
    task_data_mpi->outputs_count.emplace_back(global_max.size());
  }

  kavtorev_d_most_different_neighbor_elements_mpi::most_different_neighbor_elements_mpi testMpiTaskParallel(
      task_data_mpi);
  ASSERT_EQ(testMpiTaskParallel.ValidationImpl(), true);
  testMpiTaskParallel.PreProcessingImpl();
  testMpiTaskParallel.RunImpl();
  testMpiTaskParallel.PostProcessingImpl();

  if (world.rank() == 0) {
    std::vector<int> reference_max(1);

    auto task_data_seq = std::make_shared<ppc::core::TaskData>();
    task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vec.data()));
    task_data_seq->inputs_count.emplace_back(global_vec.size());
    task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t*>(reference_max.data()));
    task_data_seq->outputs_count.emplace_back(reference_max.size());

    kavtorev_d_most_different_neighbor_elements_mpi::most_different_neighbor_elements_seq test_task_sequential(
        task_data_seq);
    ASSERT_EQ(test_task_sequential.ValidationImpl(), true);
    test_task_sequential.PreProcessingImpl();
    test_task_sequential.RunImpl();
    test_task_sequential.PostProcessingImpl();

    ASSERT_EQ(reference_max[0], global_max[0]);
  }
}

TEST(kavtorev_d_most_different_neighbor_elements_mpi, AlternatingPositiveAndNegativeNumbers_ReturnsCorrectPair) {
  boost::mpi::communicator world;
  std::vector<int> global_vec;
  std::vector<int> global_max(1);

  auto task_data_mpi = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    global_vec = {1, -1, 2, -2, 3, -3, 4, -4, 5, -5};
    task_data_mpi->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vec.data()));
    task_data_mpi->inputs_count.emplace_back(global_vec.size());
    task_data_mpi->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_max.data()));
    task_data_mpi->outputs_count.emplace_back(global_max.size());
  }

  kavtorev_d_most_different_neighbor_elements_mpi::most_different_neighbor_elements_mpi testMpiTaskParallel(
      task_data_mpi);
  ASSERT_EQ(testMpiTaskParallel.ValidationImpl(), true);
  testMpiTaskParallel.PreProcessingImpl();
  testMpiTaskParallel.RunImpl();
  testMpiTaskParallel.PostProcessingImpl();

  if (world.rank() == 0) {
    std::vector<int> reference_max(1);

    auto task_data_seq = std::make_shared<ppc::core::TaskData>();
    task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vec.data()));
    task_data_seq->inputs_count.emplace_back(global_vec.size());
    task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t*>(reference_max.data()));
    task_data_seq->outputs_count.emplace_back(reference_max.size());

    kavtorev_d_most_different_neighbor_elements_mpi::most_different_neighbor_elements_seq test_task_sequential(
        task_data_seq);
    ASSERT_EQ(test_task_sequential.ValidationImpl(), true);
    test_task_sequential.PreProcessingImpl();
    test_task_sequential.RunImpl();
    test_task_sequential.PostProcessingImpl();

    ASSERT_EQ(reference_max[0], global_max[0]);
  }
}

TEST(kavtorev_d_most_different_neighbor_elements_mpi, LargeInputSize_ReturnsCorrectPair) {
  boost::mpi::communicator world;
  std::vector<int> global_vec;
  std::vector<int> global_max(1);

  auto task_data_mpi = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    global_vec = kavtorev_d_most_different_neighbor_elements_mpi::generator(1000);
    task_data_mpi->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vec.data()));
    task_data_mpi->inputs_count.emplace_back(global_vec.size());
    task_data_mpi->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_max.data()));
    task_data_mpi->outputs_count.emplace_back(global_max.size());
  }

  kavtorev_d_most_different_neighbor_elements_mpi::most_different_neighbor_elements_mpi testMpiTaskParallel(
      task_data_mpi);
  ASSERT_EQ(testMpiTaskParallel.ValidationImpl(), true);
  testMpiTaskParallel.PreProcessingImpl();
  testMpiTaskParallel.RunImpl();
  testMpiTaskParallel.PostProcessingImpl();

  if (world.rank() == 0) {
    std::vector<int> reference_max(1);

    auto task_data_seq = std::make_shared<ppc::core::TaskData>();
    task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vec.data()));
    task_data_seq->inputs_count.emplace_back(global_vec.size());
    task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t*>(reference_max.data()));
    task_data_seq->outputs_count.emplace_back(reference_max.size());

    kavtorev_d_most_different_neighbor_elements_mpi::most_different_neighbor_elements_seq test_task_sequential(
        task_data_seq);
    ASSERT_EQ(test_task_sequential.ValidationImpl(), true);
    test_task_sequential.PreProcessingImpl();
    test_task_sequential.RunImpl();
    test_task_sequential.PostProcessingImpl();

    ASSERT_EQ(reference_max[0], global_max[0]);
  }
}

TEST(kavtorev_d_most_different_neighbor_elements_mpi, AllNegativeNumbers_ReturnsCorrectPair) {
  boost::mpi::communicator world;
  std::vector<int> global_vec;
  std::vector<int> global_max(1);

  auto task_data_mpi = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    global_vec = {-10, -20, -30, -40, -50};
    task_data_mpi->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vec.data()));
    task_data_mpi->inputs_count.emplace_back(global_vec.size());
    task_data_mpi->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_max.data()));
    task_data_mpi->outputs_count.emplace_back(global_max.size());
  }

  kavtorev_d_most_different_neighbor_elements_mpi::most_different_neighbor_elements_mpi testMpiTaskParallel(
      task_data_mpi);
  ASSERT_EQ(testMpiTaskParallel.ValidationImpl(), true);
  testMpiTaskParallel.PreProcessingImpl();
  testMpiTaskParallel.RunImpl();
  testMpiTaskParallel.PostProcessingImpl();

  if (world.rank() == 0) {
    std::vector<int> reference_max(1);

    auto task_data_seq = std::make_shared<ppc::core::TaskData>();
    task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vec.data()));
    task_data_seq->inputs_count.emplace_back(global_vec.size());
    task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t*>(reference_max.data()));
    task_data_seq->outputs_count.emplace_back(reference_max.size());

    kavtorev_d_most_different_neighbor_elements_mpi::most_different_neighbor_elements_seq test_task_sequential(
        task_data_seq);
    ASSERT_EQ(test_task_sequential.ValidationImpl(), true);
    test_task_sequential.PreProcessingImpl();
    test_task_sequential.RunImpl();
    test_task_sequential.PostProcessingImpl();

    ASSERT_EQ(reference_max[0], global_max[0]);
  }
}

TEST(kavtorev_d_most_different_neighbor_elements_mpi, RandomNumbers_ReturnsCorrectPair) {
  boost::mpi::communicator world;
  std::vector<int> global_vec;
  std::vector<int> global_max(1);

  auto task_data_mpi = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    global_vec = kavtorev_d_most_different_neighbor_elements_mpi::generator(10);
    task_data_mpi->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vec.data()));
    task_data_mpi->inputs_count.emplace_back(global_vec.size());
    task_data_mpi->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_max.data()));
    task_data_mpi->outputs_count.emplace_back(global_max.size());
  }

  kavtorev_d_most_different_neighbor_elements_mpi::most_different_neighbor_elements_mpi testMpiTaskParallel(
      task_data_mpi);
  ASSERT_EQ(testMpiTaskParallel.ValidationImpl(), true);
  testMpiTaskParallel.PreProcessingImpl();
  testMpiTaskParallel.RunImpl();
  testMpiTaskParallel.PostProcessingImpl();

  if (world.rank() == 0) {
    std::vector<int> reference_max(1);

    auto task_data_seq = std::make_shared<ppc::core::TaskData>();
    task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vec.data()));
    task_data_seq->inputs_count.emplace_back(global_vec.size());
    task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t*>(reference_max.data()));
    task_data_seq->outputs_count.emplace_back(reference_max.size());

    kavtorev_d_most_different_neighbor_elements_mpi::most_different_neighbor_elements_seq test_task_sequential(
        task_data_seq);
    ASSERT_EQ(test_task_sequential.ValidationImpl(), true);
    test_task_sequential.PreProcessingImpl();
    test_task_sequential.RunImpl();
    test_task_sequential.PostProcessingImpl();

    ASSERT_EQ(reference_max[0], global_max[0]);
  }
}

TEST(kavtorev_d_most_different_neighbor_elements_mpi, RepeatedNumbers_ReturnsCorrectPair) {
  boost::mpi::communicator world;
  std::vector<int> global_vec;
  std::vector<int> global_max(1);

  auto task_data_mpi = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    global_vec = {5, 5, 5, 5, 5};
    task_data_mpi->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vec.data()));
    task_data_mpi->inputs_count.emplace_back(global_vec.size());
    task_data_mpi->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_max.data()));
    task_data_mpi->outputs_count.emplace_back(global_max.size());
  }

  kavtorev_d_most_different_neighbor_elements_mpi::most_different_neighbor_elements_mpi testMpiTaskParallel(
      task_data_mpi);
  ASSERT_EQ(testMpiTaskParallel.ValidationImpl(), true);
  testMpiTaskParallel.PreProcessingImpl();
  testMpiTaskParallel.RunImpl();
  testMpiTaskParallel.PostProcessingImpl();

  if (world.rank() == 0) {
    std::vector<int> reference_max(1);

    auto task_data_seq = std::make_shared<ppc::core::TaskData>();
    task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vec.data()));
    task_data_seq->inputs_count.emplace_back(global_vec.size());
    task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t*>(reference_max.data()));
    task_data_seq->outputs_count.emplace_back(reference_max.size());

    kavtorev_d_most_different_neighbor_elements_mpi::most_different_neighbor_elements_seq test_task_sequential(
        task_data_seq);
    ASSERT_EQ(test_task_sequential.ValidationImpl(), true);
    test_task_sequential.PreProcessingImpl();
    test_task_sequential.RunImpl();
    test_task_sequential.PostProcessingImpl();

    ASSERT_EQ(reference_max[0], global_max[0]);
  }
}

TEST(kavtorev_d_most_different_neighbor_elements_mpi, LargeRangeNumbers_ReturnsCorrectPair) {
  boost::mpi::communicator world;
  std::vector<int> global_vec;
  std::vector<int> global_max(1);

  auto task_data_mpi = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    global_vec = {-1000000, 1000000};
    task_data_mpi->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vec.data()));
    task_data_mpi->inputs_count.emplace_back(global_vec.size());
    task_data_mpi->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_max.data()));
    task_data_mpi->outputs_count.emplace_back(global_max.size());
  }

  kavtorev_d_most_different_neighbor_elements_mpi::most_different_neighbor_elements_mpi testMpiTaskParallel(
      task_data_mpi);
  ASSERT_EQ(testMpiTaskParallel.ValidationImpl(), true);
  testMpiTaskParallel.PreProcessingImpl();
  testMpiTaskParallel.RunImpl();
  testMpiTaskParallel.PostProcessingImpl();

  if (world.rank() == 0) {
    std::vector<int> reference_max(1);

    auto task_data_seq = std::make_shared<ppc::core::TaskData>();
    task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vec.data()));
    task_data_seq->inputs_count.emplace_back(global_vec.size());
    task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t*>(reference_max.data()));
    task_data_seq->outputs_count.emplace_back(reference_max.size());

    kavtorev_d_most_different_neighbor_elements_mpi::most_different_neighbor_elements_seq test_task_sequential(
        task_data_seq);
    ASSERT_EQ(test_task_sequential.ValidationImpl(), true);
    test_task_sequential.PreProcessingImpl();
    test_task_sequential.RunImpl();
    test_task_sequential.PostProcessingImpl();

    ASSERT_EQ(reference_max[0], global_max[0]);
  }
}

TEST(kavtorev_d_most_different_neighbor_elements_mpi, EmptyInput_ReturnsFalse) {
  boost::mpi::communicator world;
  std::vector<int> global_vec(1);

  auto task_data_mpi = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    std::vector<int> reference_ans(1, 0);

    auto task_data_seq = std::make_shared<ppc::core::TaskData>();
    task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vec.data()));
    task_data_seq->inputs_count.emplace_back(global_vec.size());
    task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t*>(reference_ans.data()));
    task_data_seq->outputs_count.emplace_back(reference_ans.size());

    kavtorev_d_most_different_neighbor_elements_mpi::most_different_neighbor_elements_seq test_task_sequential(
        task_data_seq);
    ASSERT_EQ(test_task_sequential.ValidationImpl(), false);
  }
}

TEST(kavtorev_d_most_different_neighbor_elements_mpi, InputSizeTwo_CorrectResult) {
  boost::mpi::communicator world;
  std::vector<int> global_vec;
  std::vector<int> global_diff(1, 0);

  auto task_data_mpi = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    const int sz = 2;
    global_vec = std::vector<int>(sz, 0);
    task_data_mpi->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vec.data()));
    task_data_mpi->inputs_count.emplace_back(global_vec.size());
    task_data_mpi->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_diff.data()));
    task_data_mpi->outputs_count.emplace_back(global_diff.size());
  }

  kavtorev_d_most_different_neighbor_elements_mpi::most_different_neighbor_elements_mpi testMpiTaskParallel(
      task_data_mpi);
  ASSERT_EQ(testMpiTaskParallel.ValidationImpl(), true);
  testMpiTaskParallel.PreProcessingImpl();
  testMpiTaskParallel.RunImpl();
  testMpiTaskParallel.PostProcessingImpl();

  if (world.rank() == 0) {
    std::vector<int> reference_diff(1, 0);

    auto task_data_seq = std::make_shared<ppc::core::TaskData>();
    task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vec.data()));
    task_data_seq->inputs_count.emplace_back(global_vec.size());
    task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t*>(reference_diff.data()));
    task_data_seq->outputs_count.emplace_back(reference_diff.size());

    kavtorev_d_most_different_neighbor_elements_mpi::most_different_neighbor_elements_seq test_task_sequential(
        task_data_seq);
    ASSERT_EQ(test_task_sequential.ValidationImpl(), true);
    test_task_sequential.PreProcessingImpl();
    test_task_sequential.RunImpl();
    test_task_sequential.PostProcessingImpl();

    ASSERT_EQ(reference_diff[0], global_diff[0]);
  }
}
TEST(kavtorev_d_most_different_neighbor_elements_mpi, LargeRandomInput_CorrectResult) {
  boost::mpi::communicator world;
  std::vector<int> global_vec;
  std::vector<int> global_max(1);

  auto task_data_mpi = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    const int sz = 1234;
    global_vec = kavtorev_d_most_different_neighbor_elements_mpi::generator(sz);
    task_data_mpi->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vec.data()));
    task_data_mpi->inputs_count.emplace_back(global_vec.size());
    task_data_mpi->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_max.data()));
    task_data_mpi->outputs_count.emplace_back(global_max.size());
  }

  kavtorev_d_most_different_neighbor_elements_mpi::most_different_neighbor_elements_mpi testMpiTaskParallel(
      task_data_mpi);
  ASSERT_EQ(testMpiTaskParallel.ValidationImpl(), true);
  testMpiTaskParallel.PreProcessingImpl();
  testMpiTaskParallel.RunImpl();
  testMpiTaskParallel.PostProcessingImpl();

  if (world.rank() == 0) {
    std::vector<int> reference_max(1);

    auto task_data_seq = std::make_shared<ppc::core::TaskData>();
    task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vec.data()));
    task_data_seq->inputs_count.emplace_back(global_vec.size());
    task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t*>(reference_max.data()));
    task_data_seq->outputs_count.emplace_back(reference_max.size());

    kavtorev_d_most_different_neighbor_elements_mpi::most_different_neighbor_elements_seq test_task_sequential(
        task_data_seq);
    ASSERT_EQ(test_task_sequential.ValidationImpl(), true);
    test_task_sequential.PreProcessingImpl();
    test_task_sequential.RunImpl();
    test_task_sequential.PostProcessingImpl();

    ASSERT_EQ(reference_max[0], global_max[0]);
  }
}

TEST(kavtorev_d_most_different_neighbor_elements_mpi, MediumRandomInput_CorrectResult) {
  boost::mpi::communicator world;
  std::vector<int> global_vec;
  std::vector<int> global_max(1);

  auto task_data_mpi = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    const int sz = 120;
    global_vec = kavtorev_d_most_different_neighbor_elements_mpi::generator(sz);
    task_data_mpi->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vec.data()));
    task_data_mpi->inputs_count.emplace_back(global_vec.size());
    task_data_mpi->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_max.data()));
    task_data_mpi->outputs_count.emplace_back(global_max.size());
  }

  kavtorev_d_most_different_neighbor_elements_mpi::most_different_neighbor_elements_mpi testMpiTaskParallel(
      task_data_mpi);
  ASSERT_EQ(testMpiTaskParallel.ValidationImpl(), true);
  testMpiTaskParallel.PreProcessingImpl();
  testMpiTaskParallel.RunImpl();
  testMpiTaskParallel.PostProcessingImpl();

  if (world.rank() == 0) {
    std::vector<int> reference_max(1);

    auto task_data_seq = std::make_shared<ppc::core::TaskData>();
    task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vec.data()));
    task_data_seq->inputs_count.emplace_back(global_vec.size());
    task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t*>(reference_max.data()));
    task_data_seq->outputs_count.emplace_back(reference_max.size());

    kavtorev_d_most_different_neighbor_elements_mpi::most_different_neighbor_elements_seq test_task_sequential(
        task_data_seq);
    ASSERT_EQ(test_task_sequential.ValidationImpl(), true);
    test_task_sequential.PreProcessingImpl();
    test_task_sequential.RunImpl();
    test_task_sequential.PostProcessingImpl();

    ASSERT_EQ(reference_max[0], global_max[0]);
  }
}

TEST(kavtorev_d_most_different_neighbor_elements_mpi, AllEqualElements_CorrectResult) {
  boost::mpi::communicator world;
  std::vector<int> global_vec;
  std::vector<int> global_max(1);
  auto task_data_mpi = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    const int sz = 100;
    global_vec = std::vector<int>(sz, 0);
    task_data_mpi->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vec.data()));
    task_data_mpi->inputs_count.emplace_back(global_vec.size());
    task_data_mpi->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_max.data()));
    task_data_mpi->outputs_count.emplace_back(global_max.size());
  }

  kavtorev_d_most_different_neighbor_elements_mpi::most_different_neighbor_elements_mpi testMpiTaskParallel(
      task_data_mpi);
  ASSERT_EQ(testMpiTaskParallel.ValidationImpl(), true);
  testMpiTaskParallel.PreProcessingImpl();
  testMpiTaskParallel.RunImpl();
  testMpiTaskParallel.PostProcessingImpl();

  if (world.rank() == 0) {
    std::vector<int> reference_max(1);

    auto task_data_seq = std::make_shared<ppc::core::TaskData>();
    task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vec.data()));
    task_data_seq->inputs_count.emplace_back(global_vec.size());
    task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t*>(reference_max.data()));
    task_data_seq->outputs_count.emplace_back(reference_max.size());

    kavtorev_d_most_different_neighbor_elements_mpi::most_different_neighbor_elements_seq test_task_sequential(
        task_data_seq);
    ASSERT_EQ(test_task_sequential.ValidationImpl(), true);
    test_task_sequential.PreProcessingImpl();
    test_task_sequential.RunImpl();
    test_task_sequential.PostProcessingImpl();

    ASSERT_EQ(reference_max[0], global_max[0]);
  }
}

TEST(kavtorev_d_most_different_neighbor_elements_mpi, AlternatingElements_CorrectResult) {
  boost::mpi::communicator world;
  std::vector<int> global_vec;
  std::vector<int> global_max(1);
  auto task_data_mpi = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    global_vec = {1, -1, 1, -1, 1, -1, 1, -1, 1, -1};
    task_data_mpi->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vec.data()));
    task_data_mpi->inputs_count.emplace_back(global_vec.size());
    task_data_mpi->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_max.data()));
    task_data_mpi->outputs_count.emplace_back(global_max.size());
  }

  kavtorev_d_most_different_neighbor_elements_mpi::most_different_neighbor_elements_mpi testMpiTaskParallel(
      task_data_mpi);
  ASSERT_EQ(testMpiTaskParallel.ValidationImpl(), true);
  testMpiTaskParallel.PreProcessingImpl();
  testMpiTaskParallel.RunImpl();
  testMpiTaskParallel.PostProcessingImpl();

  if (world.rank() == 0) {
    std::vector<int> reference_max(1);

    auto task_data_seq = std::make_shared<ppc::core::TaskData>();
    task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vec.data()));
    task_data_seq->inputs_count.emplace_back(global_vec.size());
    task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t*>(reference_max.data()));
    task_data_seq->outputs_count.emplace_back(reference_max.size());

    kavtorev_d_most_different_neighbor_elements_mpi::most_different_neighbor_elements_seq test_task_sequential(
        task_data_seq);
    ASSERT_EQ(test_task_sequential.ValidationImpl(), true);
    test_task_sequential.PreProcessingImpl();
    test_task_sequential.RunImpl();
    test_task_sequential.PostProcessingImpl();

    ASSERT_EQ(reference_max[0], global_max[0]);
  }
}

TEST(kavtorev_d_most_different_neighbor_elements_mpi, ConstantDifferenceSequence_CorrectResult) {
  boost::mpi::communicator world;
  std::vector<int> global_vec;
  std::vector<int> global_max(1);
  auto task_data_mpi = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    const int sz = 123;
    global_vec.resize(sz);
    for (int i = 0; i < sz; ++i) {
      global_vec[i] = sz - i;
    }
    task_data_mpi->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vec.data()));
    task_data_mpi->inputs_count.emplace_back(global_vec.size());
    task_data_mpi->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_max.data()));
    task_data_mpi->outputs_count.emplace_back(global_max.size());
  }

  kavtorev_d_most_different_neighbor_elements_mpi::most_different_neighbor_elements_mpi testMpiTaskParallel(
      task_data_mpi);
  ASSERT_EQ(testMpiTaskParallel.ValidationImpl(), true);
  testMpiTaskParallel.PreProcessingImpl();
  testMpiTaskParallel.RunImpl();
  testMpiTaskParallel.PostProcessingImpl();

  if (world.rank() == 0) {
    std::vector<int> reference_max(1);

    auto task_data_seq = std::make_shared<ppc::core::TaskData>();
    task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vec.data()));
    task_data_seq->inputs_count.emplace_back(global_vec.size());
    task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t*>(reference_max.data()));
    task_data_seq->outputs_count.emplace_back(reference_max.size());

    kavtorev_d_most_different_neighbor_elements_mpi::most_different_neighbor_elements_seq test_task_sequential(
        task_data_seq);
    ASSERT_EQ(test_task_sequential.ValidationImpl(), true);
    test_task_sequential.PreProcessingImpl();
    test_task_sequential.RunImpl();
    test_task_sequential.PostProcessingImpl();

    ASSERT_EQ(reference_max[0], global_max[0]);
  }
}

TEST(kavtorev_d_most_different_neighbor_elements_mpi, MostlyZerosInput_ReturnsCorrectPair) {
  boost::mpi::communicator world;
  std::vector<int> global_vec;
  std::vector<int> global_max(1);
  auto task_data_mpi = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    global_vec = {12, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    task_data_mpi->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vec.data()));
    task_data_mpi->inputs_count.emplace_back(global_vec.size());
    task_data_mpi->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_max.data()));
    task_data_mpi->outputs_count.emplace_back(global_max.size());
  }

  kavtorev_d_most_different_neighbor_elements_mpi::most_different_neighbor_elements_mpi testMpiTaskParallel(
      task_data_mpi);
  ASSERT_EQ(testMpiTaskParallel.ValidationImpl(), true);
  testMpiTaskParallel.PreProcessingImpl();
  testMpiTaskParallel.RunImpl();
  testMpiTaskParallel.PostProcessingImpl();

  if (world.rank() == 0) {
    std::vector<int> reference_max(1);

    auto task_data_seq = std::make_shared<ppc::core::TaskData>();
    task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vec.data()));
    task_data_seq->inputs_count.emplace_back(global_vec.size());
    task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t*>(reference_max.data()));
    task_data_seq->outputs_count.emplace_back(reference_max.size());

    kavtorev_d_most_different_neighbor_elements_mpi::most_different_neighbor_elements_seq test_task_sequential(
        task_data_seq);
    ASSERT_EQ(test_task_sequential.ValidationImpl(), true);
    test_task_sequential.PreProcessingImpl();
    test_task_sequential.RunImpl();
    test_task_sequential.PostProcessingImpl();

    ASSERT_EQ(reference_max[0], global_max[0]);
  }
}