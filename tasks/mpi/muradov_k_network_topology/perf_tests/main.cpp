#define OMPI_SKIP_MPICXX 1

#include <gtest/gtest.h>
#include <mpi.h>

#include <vector>

#include "mpi/muradov_k_network_topology/include/ops_mpi.hpp"

namespace muradov_k_network_topology_mpi {

TEST(muradov_k_network_topology_mpi, test_pipeline_run) {
  MPI_Comm comm = MPI_COMM_WORLD;
  int size = 0;
  int rank = 0;
  MPI_Comm_size(comm, &size);
  MPI_Comm_rank(comm, &rank);
  if (size < 2) {
    GTEST_SKIP();
  }

  NetworkTopology topology(comm);
  topology.CreateRingTopology();

  constexpr int kIterations = 100;
  constexpr int kMessageSize = 1024 * 1024;  // 1 MB
  std::vector<char> buffer(kMessageSize, static_cast<char>(rank));

  for (int i = 0; i < kIterations; ++i) {
    std::vector<char> recv_buffer(kMessageSize, 0);
    ASSERT_TRUE(topology.RingExchange(buffer.data(), recv_buffer.data(), kMessageSize, MPI_BYTE));
  }

  MPI_Barrier(comm);
  SUCCEED();
}

TEST(muradov_k_network_topology_mpi, test_task_run) {
  MPI_Comm comm = MPI_COMM_WORLD;
  int size = 0;
  int rank = 0;
  MPI_Comm_size(comm, &size);
  MPI_Comm_rank(comm, &rank);
  if (size < 2) {
    GTEST_SKIP();
  }

  NetworkTopology topology(comm);
  topology.CreateRingTopology();

  constexpr int kMessageSize = 1024;  // 1 KB message
  std::vector<char> buffer(kMessageSize, static_cast<char>(rank));
  std::vector<char> recv_buffer(kMessageSize, 0);

  ASSERT_TRUE(topology.RingExchange(buffer.data(), recv_buffer.data(), kMessageSize, MPI_BYTE));

  MPI_Barrier(comm);
  SUCCEED();
}

}  // namespace muradov_k_network_topology_mpi
