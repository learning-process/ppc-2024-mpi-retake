#define OMPI_SKIP_MPICXX 1

#include <gtest/gtest.h>
#include <mpi.h>

#include "mpi/muradov_k_network_topology/include/ops_mpi.hpp"

namespace muradov_k_network_topology_mpi {

TEST(muradov_k_network_topology_mpi, FullRingCommunication) {
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

  int test_data = rank * 100;
  int received_data = -1;
  ASSERT_TRUE(topology.RingExchange(&test_data, &received_data, 1, MPI_INT));

  int expected = ((rank - 1 + size) % size) * 100;
  ASSERT_EQ(received_data, expected);
}

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

TEST(muradov_k_network_topology_mpi, ReceiveWithoutTopology) {
  MPI_Comm comm = MPI_COMM_WORLD;
  int rank = 0;
  MPI_Comm_rank(comm, &rank);

  NetworkTopology topology(comm);
  int data = -1;
  bool result = topology.Receive(MPI_ANY_SOURCE, &data, 1, MPI_INT);
  ASSERT_FALSE(result);
}

TEST(muradov_k_network_topology_mpi, MultipleRoundCommunication) {
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

  int value = rank;
  const int rounds = size;
  int temp = 0;
  for (int i = 0; i < rounds; ++i) {
    ASSERT_TRUE(topology.RingExchange(&value, &temp, 1, MPI_INT));
    value = temp;
  }
  ASSERT_EQ(value, rank);
}

TEST(muradov_k_network_topology_mpi, AnySourceReceive) {
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

  int message = rank + 500;
  int received = -1;

  if (rank != 0 && rank != 1) {
    GTEST_SKIP();
  }

  if (rank == 0) {
    ASSERT_TRUE(topology.Send(1, &message, 1, MPI_INT));
  } else if (rank == 1) {
    ASSERT_TRUE(topology.Receive(MPI_ANY_SOURCE, &received, 1, MPI_INT));
    ASSERT_EQ(received, 500);
  }
}

}  // namespace muradov_k_network_topology_mpi
