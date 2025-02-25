#define OMPI_SKIP_MPICXX
#include <gtest/gtest.h>
#include <mpi.h>

#include <vector>

#include "mpi/muradov_k_network_topology/include/ops_mpi.hpp"

namespace muradov_k_network_topology_mpi {

// Test 1: Full ring communication between neighboring processes.
TEST(muradov_k_network_topology_mpi, FullRingCommunication) {
  MPI_Comm comm = MPI_COMM_WORLD;
  int size = 0, rank = 0;
  MPI_Comm_size(comm, &size);
  MPI_Comm_rank(comm, &rank);
  if (size < 2) {
    GTEST_SKIP() << "Requires at least 2 processes";
  }

  NetworkTopology topology(comm);
  topology.CreateRingTopology();

  int test_data = rank * 100;
  int received_data = -1;
  const int target = (rank + 1) % size;
  const int source = (rank - 1 + size) % size;

  if (rank % 2 == 0) {
    ASSERT_TRUE(topology.Send(target, &test_data, 1, MPI_INT));
    ASSERT_TRUE(topology.Receive(source, &received_data, 1, MPI_INT));
  } else {
    ASSERT_TRUE(topology.Receive(source, &received_data, 1, MPI_INT));
    ASSERT_TRUE(topology.Send(target, &test_data, 1, MPI_INT));
  }

  const int expected = (source * 100);
  ASSERT_EQ(received_data, expected);
}

// Test 2: Calling Send without first creating the topology should fail.
TEST(muradov_k_network_topology_mpi, SendWithoutTopology) {
  MPI_Comm comm = MPI_COMM_WORLD;
  int rank = 0;
  MPI_Comm_rank(comm, &rank);

  NetworkTopology topology(comm);
  // Do not call CreateRingTopology.
  int data = rank;
  bool result = topology.Send((rank + 1) % 2, &data, 1, MPI_INT);
  ASSERT_FALSE(result);
}

// Test 3: Calling Receive without first creating the topology should fail.
TEST(muradov_k_network_topology_mpi, ReceiveWithoutTopology) {
  MPI_Comm comm = MPI_COMM_WORLD;
  int rank = 0;
  MPI_Comm_rank(comm, &rank);

  NetworkTopology topology(comm);
  // Do not call CreateRingTopology.
  int data = -1;
  bool result = topology.Receive(MPI_ANY_SOURCE, &data, 1, MPI_INT);
  ASSERT_FALSE(result);
}

// Test 4: Multiple rounds of ring communication should return the original value.
TEST(muradov_k_network_topology_mpi, MultipleRoundCommunication) {
  MPI_Comm comm = MPI_COMM_WORLD;
  int size = 0, rank = 0;
  MPI_Comm_size(comm, &size);
  MPI_Comm_rank(comm, &rank);
  if (size < 2) {
    GTEST_SKIP() << "Requires at least 2 processes";
  }

  NetworkTopology topology(comm);
  topology.CreateRingTopology();

  // Each process starts with its rank.
  int value = rank;
  const int rounds = size;  // After 'size' rounds, each message should return to its origin.
  for (int i = 0; i < rounds; ++i) {
    int send_value = value;
    int recv_value = -1;
    const int target = (rank + 1) % size;
    const int source = (rank - 1 + size) % size;
    if (rank % 2 == 0) {
      ASSERT_TRUE(topology.Send(target, &send_value, 1, MPI_INT));
      ASSERT_TRUE(topology.Receive(source, &recv_value, 1, MPI_INT));
    } else {
      ASSERT_TRUE(topology.Receive(source, &recv_value, 1, MPI_INT));
      ASSERT_TRUE(topology.Send(target, &send_value, 1, MPI_INT));
    }
    value = recv_value;
  }
  // After a full rotation, each process should have its original value.
  ASSERT_EQ(value, rank);
}

// Test 5: Verify that using MPI_ANY_SOURCE in Receive works correctly.
TEST(muradov_k_network_topology_mpi, AnySourceReceive) {
  MPI_Comm comm = MPI_COMM_WORLD;
  int size = 0, rank = 0;
  MPI_Comm_size(comm, &size);
  MPI_Comm_rank(comm, &rank);
  if (size < 2) {
    GTEST_SKIP() << "Requires at least 2 processes";
  }

  NetworkTopology topology(comm);
  topology.CreateRingTopology();

  int message = rank + 500;
  int received = -1;

  // For clarity, only processes 0 and 1 are involved.
  if (rank == 0) {
    ASSERT_TRUE(topology.Send(1, &message, 1, MPI_INT));
  } else if (rank == 1) {
    ASSERT_TRUE(topology.Receive(MPI_ANY_SOURCE, &received, 1, MPI_INT));
    // Since process 0 sends message = 0 + 500.
    ASSERT_EQ(received, 500);
  } else {
    GTEST_SKIP() << "Test applicable only to processes 0 and 1";
  }
}

}  // namespace muradov_k_network_topology_mpi
