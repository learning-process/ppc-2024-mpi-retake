#include <gtest/gtest.h>

#include <boost/mpi/communicator.hpp>
#include <vector>
#include <string>
#include <cstdlib>
#include <cstdint>
#include <memory>

#include "core/task/include/task.hpp"
#include "mpi/chernova_n_topology_ring/include/ops_mpi.hpp"

namespace {

std::vector<char> GenerateData(int k) {
  const std::string words[] = {"one", "two", "three"};

  std::string result;
  size_t j = words->size();

  for (int i = 0; i < k; ++i) {
    result += words[rand() % (j)];
    if (i < k - 1) {
      result += ' ';
    }
  }

  return {result.begin(), result.end()};
}

}  // namespace

TEST(chernova_n_topology_ring_mpi, test_empty_string) {
  boost::mpi::communicator world;
  std::vector<char> in = {};
  const int n = static_cast<int>(in.size());
  std::vector<char> out_vec(n);
  std::vector<int> out_process;
  auto task_data_parallel = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    out_process = std::vector<int>(world.size() + 1);
    task_data_parallel->inputs.emplace_back(reinterpret_cast<uint8_t*>(const_cast<char*>(in.data())));
    task_data_parallel->inputs_count.emplace_back(in.size());
    task_data_parallel->outputs.emplace_back(reinterpret_cast<uint8_t*>(out_vec.data()));
    task_data_parallel->outputs.emplace_back(reinterpret_cast<uint8_t*>(out_process.data()));
    task_data_parallel->outputs_count.emplace_back(2);
  }
  chernova_n_topology_ring_mpi::TestMPITaskParallel test_task_parallel(task_data_parallel);
  if (world.rank() == 0) {
    ASSERT_EQ(test_task_parallel.Validation(), false);
  }
}

TEST(chernova_n_topology_ring_mpi, test_ten_symbols) {
  boost::mpi::communicator world;
  std::vector<char> in = {'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j'};
  const int n = static_cast<int>(in.size());
  std::vector<char> out_vec(n);
  std::vector<int> out_process;
  auto task_data_parallel = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    out_process = std::vector<int>(world.size() + 1);
    task_data_parallel->inputs.emplace_back(reinterpret_cast<uint8_t*>(const_cast<char*>(in.data())));
    task_data_parallel->inputs_count.emplace_back(in.size());
    task_data_parallel->outputs.emplace_back(reinterpret_cast<uint8_t*>(out_vec.data()));
    task_data_parallel->outputs.emplace_back(reinterpret_cast<uint8_t*>(out_process.data()));
    task_data_parallel->outputs_count.reserve(2);
  }
  chernova_n_topology_ring_mpi::TestMPITaskParallel test_task_parallel(task_data_parallel);
  ASSERT_EQ(test_task_parallel.Validation(), true);
  test_task_parallel.PreProcessing();
  test_task_parallel.Run();
  test_task_parallel.PostProcessing();
  if (world.rank() == 0) {
    if (world.size() != 1) {
      for (int i = 0; i != world.size(); ++i) {
        EXPECT_EQ(i, out_process[i]);
      }
    }
    ASSERT_EQ(true, std::ranges::equal(in, out_vec));
  }
}

TEST(chernova_n_topology_ring_mpi, test_five_words) {
  boost::mpi::communicator world;
  std::vector<char> in = GenerateData(5);
  const int n = static_cast<int>(in.size());
  std::vector<char> out_vec(n);
  std::vector<int> out_process;
  auto task_data_parallel = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    out_process = std::vector<int>(world.size() + 1);
    task_data_parallel->inputs.emplace_back(reinterpret_cast<uint8_t*>(const_cast<char*>(in.data())));
    task_data_parallel->inputs_count.emplace_back(in.size());
    task_data_parallel->outputs.emplace_back(reinterpret_cast<uint8_t*>(out_vec.data()));
    task_data_parallel->outputs.emplace_back(reinterpret_cast<uint8_t*>(out_process.data()));
    task_data_parallel->outputs_count.reserve(2);
  }
  chernova_n_topology_ring_mpi::TestMPITaskParallel test_task_parallel(task_data_parallel);
  ASSERT_EQ(test_task_parallel.Validation(), true);
  test_task_parallel.PreProcessing();
  test_task_parallel.Run();
  test_task_parallel.PostProcessing();
  if (world.rank() == 0) {
    if (world.size() != 1) {
      for (int i = 0; i != world.size(); ++i) {
        EXPECT_EQ(i, out_process[i]);
      }
    }
    ASSERT_EQ(true, std::ranges::equal(in, out_vec));
  }
}

TEST(chernova_n_topology_ring_mpi, test_ten_words) {
  boost::mpi::communicator world;
  std::vector<char> in = GenerateData(10);
  const int n = static_cast<int>(in.size());
  std::vector<char> out_vec(n);
  std::vector<int> out_process;
  auto task_data_parallel = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    out_process = std::vector<int>(world.size() + 1);
    task_data_parallel->inputs.emplace_back(reinterpret_cast<uint8_t*>(const_cast<char*>(in.data())));
    task_data_parallel->inputs_count.emplace_back(in.size());
    task_data_parallel->outputs.emplace_back(reinterpret_cast<uint8_t*>(out_vec.data()));
    task_data_parallel->outputs.emplace_back(reinterpret_cast<uint8_t*>(out_process.data()));
    task_data_parallel->outputs_count.reserve(2);
  }
  chernova_n_topology_ring_mpi::TestMPITaskParallel test_task_parallel(task_data_parallel);
  ASSERT_EQ(test_task_parallel.Validation(), true);
  test_task_parallel.PreProcessing();
  test_task_parallel.Run();
  test_task_parallel.PostProcessing();
  if (world.rank() == 0) {
    if (world.size() != 1) {
      for (int i = 0; i != world.size(); ++i) {
        EXPECT_EQ(i, out_process[i]);
      }
    }
    ASSERT_EQ(true, std::ranges::equal(in, out_vec));
  }
}

TEST(chernova_n_topology_ring_mpi, test_twenty_words) {
  boost::mpi::communicator world;
  std::vector<char> in = GenerateData(20);
  const int n = static_cast<int>(in.size());
  std::vector<char> out_vec(n);
  std::vector<int> out_process;
  auto task_data_parallel = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    out_process = std::vector<int>(world.size() + 1);
    task_data_parallel->inputs.emplace_back(reinterpret_cast<uint8_t*>(const_cast<char*>(in.data())));
    task_data_parallel->inputs_count.emplace_back(in.size());
    task_data_parallel->outputs.emplace_back(reinterpret_cast<uint8_t*>(out_vec.data()));
    task_data_parallel->outputs.emplace_back(reinterpret_cast<uint8_t*>(out_process.data()));
    task_data_parallel->outputs_count.reserve(2);
  }
  chernova_n_topology_ring_mpi::TestMPITaskParallel test_task_parallel(task_data_parallel);
  ASSERT_EQ(test_task_parallel.Validation(), true);
  test_task_parallel.PreProcessing();
  test_task_parallel.Run();
  test_task_parallel.PostProcessing();
  if (world.rank() == 0) {
    if (world.size() != 1) {
      for (int i = 0; i != world.size(); ++i) {
        EXPECT_EQ(i, out_process[i]);
      }
    }
    ASSERT_EQ(true, std::ranges::equal(in, out_vec));
  }
}

TEST(chernova_n_topology_ring_mpi, test_thirty_words) {
  boost::mpi::communicator world;
  std::vector<char> in = GenerateData(30);
  const int n = static_cast<int>(in.size());
  std::vector<char> out_vec(n);
  std::vector<int> out_process;
  auto task_data_parallel = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    out_process = std::vector<int>(world.size() + 1);
    task_data_parallel->inputs.emplace_back(reinterpret_cast<uint8_t*>(const_cast<char*>(in.data())));
    task_data_parallel->inputs_count.emplace_back(in.size());
    task_data_parallel->outputs.emplace_back(reinterpret_cast<uint8_t*>(out_vec.data()));
    task_data_parallel->outputs.emplace_back(reinterpret_cast<uint8_t*>(out_process.data()));
    task_data_parallel->outputs_count.reserve(2);
  }
  chernova_n_topology_ring_mpi::TestMPITaskParallel test_task_parallel(task_data_parallel);
  ASSERT_EQ(test_task_parallel.Validation(), true);
  test_task_parallel.PreProcessing();
  test_task_parallel.Run();
  test_task_parallel.PostProcessing();
  if (world.rank() == 0) {
    if (world.size() != 1) {
      for (int i = 0; i != world.size(); ++i) {
        EXPECT_EQ(i, out_process[i]);
      }
    }
    ASSERT_EQ(true, std::ranges::equal(in, out_vec));
  }
}