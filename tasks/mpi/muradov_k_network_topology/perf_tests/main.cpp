#define OMPI_SKIP_MPICXX
#include <gtest/gtest.h>
#include <mpi.h>

#include <chrono>
#include <iostream>
#include <vector>

#include "mpi/muradov_k_network_topology/include/ops_mpi.hpp"

namespace muradov_k_network_topology_mpi {

TEST(muradov_k_network_topology_mpi, RingBandwidthMeasurement) {
  MPI_Comm comm = MPI_COMM_WORLD;
  NetworkTopology topology(comm);
  topology.CreateRingTopology();

  int rank = 0;
  int size = 0;
  MPI_Comm_rank(comm, &rank);
  MPI_Comm_size(comm, &size);

  const int iterations = 100;
  const int message_size = 1024 * 1024;  // 1MB
  // Cast rank to char to avoid narrowing conversion warning.
  std::vector<char> buffer(message_size, static_cast<char>(rank));
  double total_time = 0.0;

  if (rank == 0) {
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iterations; ++i) {
      topology.Send(1, buffer.data(), message_size, MPI_BYTE);
    }
    auto end = std::chrono::high_resolution_clock::now();
    total_time = std::chrono::duration<double>(end - start).count();
    std::cout << "Bandwidth: " << (message_size * iterations * 8) / (total_time * 1e6) << " Mbps" << "\n";
  } else if (rank == 1) {
    for (int i = 0; i < iterations; ++i) {
      topology.Receive(0, buffer.data(), message_size, MPI_BYTE);
    }
  }

  MPI_Barrier(comm);
  SUCCEED();
}

}  // namespace muradov_k_network_topology_mpi
