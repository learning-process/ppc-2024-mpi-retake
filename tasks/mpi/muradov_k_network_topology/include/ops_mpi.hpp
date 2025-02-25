#ifndef MURADOV_K_NETWORK_TOPOLOGY_OPS_MPI_HPP
#define MURADOV_K_NETWORK_TOPOLOGY_OPS_MPI_HPP

#include <mpi.h>

namespace muradov_k_network_topology_mpi {

class NetworkTopology {
 public:
  explicit NetworkTopology(MPI_Comm global_comm);
  ~NetworkTopology();

  // Renamed to follow PascalCase per clang-tidy recommendations.
  void CreateRingTopology();
  bool Send(int dest, const void* data, int count, MPI_Datatype datatype);
  bool Receive(int source, void* buffer, int count, MPI_Datatype datatype);

 private:
  MPI_Comm global_comm_;
  MPI_Comm topology_comm_;
  int rank_;
  int size_;
  int left_;
  int right_;
};

}  // namespace muradov_k_network_topology_mpi

#endif  // MURADOV_K_NETWORK_TOPOLOGY_OPS_MPI_HPP
