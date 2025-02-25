#define OMPI_SKIP_MPICXX
#include <gtest/gtest.h>
#include <mpi.h>

#include <chrono>
#include <iostream>
#include <vector>

#include "mpi/muradov_k_network_topology/include/ops_mpi.hpp"

namespace muradov_k_network_topology_mpi {

// Performance Test 1: test_pipeline_run – measure the bandwidth over multiple iterations.
TEST(muradov_k_network_topology_mpi, test_pipeline_run) {
  MPI_Comm comm = MPI_COMM_WORLD;
  int size = 0, rank = 0;
  MPI_Comm_size(comm, &size);
  MPI_Comm_rank(comm, &rank);
  if (size < 2) {
    GTEST_SKIP() << "Requires at least 2 processes";
  }

  NetworkTopology topology(comm);
  topology.CreateRingTopology();

  constexpr int iterations = 100;
  constexpr int message_size = 1024 * 1024;  // 1 MB
  std::vector<char> buffer(message_size, static_cast<char>(rank));
  double total_time = 0.0;

  if (rank == 0) {
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iterations; ++i) {
      ASSERT_TRUE(topology.Send(1, buffer.data(), message_size, MPI_BYTE));
    }
    auto end = std::chrono::high_resolution_clock::now();
    total_time = std::chrono::duration<double>(end - start).count();
    double bandwidth = (message_size * iterations * 8) / (total_time * 1e6);  // Mbps
    std::cout << "test_pipeline_run - Bandwidth: " << bandwidth << " Mbps\n";
  } else if (rank == 1) {
    for (int i = 0; i < iterations; ++i) {
      ASSERT_TRUE(topology.Receive(0, buffer.data(), message_size, MPI_BYTE));
    }
  }

  MPI_Barrier(comm);
  SUCCEED();
}

// Performance Test 2: test_task_run – measure the round-trip time for a single communication.
TEST(muradov_k_network_topology_mpi, test_task_run) {
  MPI_Comm comm = MPI_COMM_WORLD;
  int size = 0, rank = 0;
  MPI_Comm_size(comm, &size);
  MPI_Comm_rank(comm, &rank);
  if (size < 2) {
    GTEST_SKIP() << "Requires at least 2 processes";
  }

  NetworkTopology topology(comm);
  topology.CreateRingTopology();

  constexpr int message_size = 1024;  // 1 KB message
  std::vector<char> buffer(message_size, static_cast<char>(rank));

  double start_time = MPI_Wtime();
  if (rank == 0) {
    ASSERT_TRUE(topology.Send(1, buffer.data(), message_size, MPI_BYTE));
    ASSERT_TRUE(topology.Receive(1, buffer.data(), message_size, MPI_BYTE));
  } else if (rank == 1) {
    ASSERT_TRUE(topology.Receive(0, buffer.data(), message_size, MPI_BYTE));
    ASSERT_TRUE(topology.Send(0, buffer.data(), message_size, MPI_BYTE));
  }
  double end_time = MPI_Wtime();
  double elapsed = end_time - start_time;
  if (rank == 0) {
    std::cout << "test_task_run - Round Trip Time: " << elapsed << " seconds\n";
  }
  MPI_Barrier(comm);
  SUCCEED();
}

}  // namespace muradov_k_network_topology_mpi
