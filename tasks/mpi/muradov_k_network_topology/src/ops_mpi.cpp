#include "mpi/muradov_k_network_topology/include/ops_mpi.hpp"

#include <stdexcept>

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

void NetworkTopology::create_ring_topology() {
  MPI_Group world_group;
  MPI_Comm_group(global_comm_, &world_group);

  left_ = (rank_ - 1 + size_) % size_;
  right_ = (rank_ + 1) % size_;

  MPI_Comm_create(global_comm_, world_group, &topology_comm_);
  MPI_Group_free(&world_group);
}

bool NetworkTopology::send(int dest, const void* data, int count, MPI_Datatype datatype) {
  if (topology_comm_ == MPI_COMM_NULL) return false;

  int current = rank_;
  while (current != dest) {
    int next = (dest > current) ? right_ : left_;
    MPI_Send(data, count, datatype, next, 0, topology_comm_);
    current = next;
  }
  return true;
}

bool NetworkTopology::receive(int source, void* buffer, int count, MPI_Datatype datatype) {
  if (topology_comm_ == MPI_COMM_NULL) return false;

  MPI_Status status;
  int flag;
  MPI_Iprobe(MPI_ANY_SOURCE, MPI_ANY_TAG, topology_comm_, &flag, &status);

  if (flag && (source == MPI_ANY_SOURCE || status.MPI_SOURCE == source)) {
    MPI_Recv(buffer, count, datatype, status.MPI_SOURCE, status.MPI_TAG, topology_comm_, MPI_STATUS_IGNORE);
    return true;
  }
  return false;
}

}  // namespace muradov_k_network_topology_mpi