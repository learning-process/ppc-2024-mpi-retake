#include <gtest/gtest.h>
#include <mpi.h>

#include "mpi/muradov_k_network_topology/include/ops_mpi.hpp"

namespace muradov_k_network_topology_mpi {

TEST(muradov_k_network_topology_mpi, FullRingCommunication) {
  MPI_Comm comm = MPI_COMM_WORLD;
  NetworkTopology topology(comm);
  topology.create_ring_topology();

  int rank, size;
  MPI_Comm_rank(comm, &rank);
  MPI_Comm_size(comm, &size);

  const int test_data = rank * 100;
  int received_data = -1;
  const int target = (rank + 1) % size;

  if (size > 1) {
    topology.send(target, &test_data, 1, MPI_INT);
    topology.receive((rank - 1 + size) % size, &received_data, 1, MPI_INT);

    ASSERT_EQ(received_data, ((rank - 1 + size) % size) * 100);
  } else {
    ASSERT_TRUE(true);  // Skip test for single process
  }
}

}  // namespace muradov_k_network_topology_mpi