#define OMPI_SKIP_MPICXX 1

#include <gtest/gtest.h>
#include <mpi.h>

#include "mpi/muradov_k_network_topology/include/ops_mpi.hpp"

namespace muradov_k_network_topology_mpi {

TEST(muradov_k_network_topology_mpi, FullRingCommunication) {
  MPI_Comm comm = MPI_COMM_WORLD;
  int size, rank;
  MPI_Comm_size(comm, &size);
  MPI_Comm_rank(comm, &rank);
  if (size < 2) {
    GTEST_SKIP() << "Requires at least 2 processes";
  }

  NetworkTopology topology(comm);
  topology.CreateRingTopology();

  int test_data = rank * 100;
  int received_data = -1;
  // The new RingExchange method takes care of the ordering.
  ASSERT_TRUE(topology.RingExchange(&test_data, &received_data, 1, MPI_INT));

  int expected = ((rank - 1 + size) % size) * 100;
  ASSERT_EQ(received_data, expected);
}

TEST(muradov_k_network_topology_mpi, SendWithoutTopology) {
  MPI_Comm comm = MPI_COMM_WORLD;
  int rank;
  MPI_Comm_rank(comm, &rank);

  NetworkTopology topology(comm);
  // Do not call CreateRingTopology.
  int data = rank;
  bool result = topology.Send((rank + 1) % 2, &data, 1, MPI_INT);
  ASSERT_FALSE(result);
}

TEST(muradov_k_network_topology_mpi, ReceiveWithoutTopology) {
  MPI_Comm comm = MPI_COMM_WORLD;
  int rank;
  MPI_Comm_rank(comm, &rank);

  NetworkTopology topology(comm);
  // Do not call CreateRingTopology.
  int data = -1;
  bool result = topology.Receive(MPI_ANY_SOURCE, &data, 1, MPI_INT);
  ASSERT_FALSE(result);
}

TEST(muradov_k_network_topology_mpi, MultipleRoundCommunication) {
  MPI_Comm comm = MPI_COMM_WORLD;
  int size, rank;
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
  int temp = 0;
  for (int i = 0; i < rounds; ++i) {
    ASSERT_TRUE(topology.RingExchange(&value, &temp, 1, MPI_INT));
    value = temp;
  }
  // After a full rotation, each process should have its original value.
  ASSERT_EQ(value, rank);
}

TEST(muradov_k_network_topology_mpi, AnySourceReceive) {
  MPI_Comm comm = MPI_COMM_WORLD;
  int size, rank;
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
  if (rank == 0 || rank == 1) {
    if (rank == 0) {
      ASSERT_TRUE(topology.Send(1, &message, 1, MPI_INT));
    } else if (rank == 1) {
      ASSERT_TRUE(topology.Receive(MPI_ANY_SOURCE, &received, 1, MPI_INT));
      // Since process 0 sends message = 500.
      ASSERT_EQ(received, 500);
    }
  } else {
    GTEST_SKIP() << "Test applicable only to processes 0 and 1";
  }
}

}  // namespace muradov_k_network_topology_mpi
