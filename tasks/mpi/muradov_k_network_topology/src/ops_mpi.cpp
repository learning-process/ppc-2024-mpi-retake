#define OMPI_SKIP_MPICXX 1

#include "mpi/muradov_k_network_topology/include/ops_mpi.hpp"

#include <mpi.h>

namespace muradov_k_network_topology_mpi {

NetworkTopology::NetworkTopology(MPI_Comm global_comm) : global_comm_(global_comm), topology_comm_(MPI_COMM_NULL) {
  MPI_Comm_rank(global_comm_, &rank_);
  MPI_Comm_size(global_comm_, &size_);
}

NetworkTopology::~NetworkTopology() {
  if (topology_comm_ != MPI_COMM_NULL) {
    MPI_Comm_free(&topology_comm_);
  }
}

void NetworkTopology::CreateRingTopology() {
  MPI_Group world_group = MPI_GROUP_NULL;
  MPI_Comm_group(global_comm_, &world_group);

  left_ = (rank_ - 1 + size_) % size_;
  right_ = (rank_ + 1) % size_;

  MPI_Comm_create(global_comm_, world_group, &topology_comm_);
  MPI_Group_free(&world_group);
}

bool NetworkTopology::Send(int dest, const void* data, int count, MPI_Datatype datatype) {
  if (topology_comm_ == MPI_COMM_NULL) {
    return false;
  }

  int current = rank_;
  while (current != dest) {
    int dist_right = (dest - current + size_) % size_;
    int dist_left = (current - dest + size_) % size_;

    int next = (dist_right <= dist_left) ? right_ : left_;
    MPI_Send(data, count, datatype, next, 0, topology_comm_);

    if (next == dest) {
      break;
    }

    int opposite = (next == right_) ? left_ : right_;
    MPI_Recv(const_cast<void*>(data), count, datatype, opposite, 0, topology_comm_, MPI_STATUS_IGNORE);
    current = next;
  }
  return true;
}

bool NetworkTopology::Receive(int source, void* buffer, int count, MPI_Datatype datatype) {
  if (topology_comm_ == MPI_COMM_NULL) {
    return false;
  }

  MPI_Status status;
  MPI_Probe(MPI_ANY_SOURCE, 0, topology_comm_, &status);

  if (source == MPI_ANY_SOURCE || status.MPI_SOURCE == source) {
    MPI_Recv(buffer, count, datatype, status.MPI_SOURCE, 0, topology_comm_, MPI_STATUS_IGNORE);
    return true;
  }
  return false;
}

bool NetworkTopology::RingExchange(const void* send_data, void* recv_data, int count, MPI_Datatype datatype) {
  if (topology_comm_ == MPI_COMM_NULL) {
    return false;
  }
  // In a ring, send to the right neighbor and receive from the left neighbor.
  MPI_Status status;
  MPI_Sendrecv(send_data, count, datatype, right_, 0, recv_data, count, datatype, left_, 0, topology_comm_, &status);
  return true;
}

}  // namespace muradov_k_network_topology_mpi
