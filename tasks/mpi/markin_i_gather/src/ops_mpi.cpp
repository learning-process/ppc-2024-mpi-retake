#include "mpi/markin_i_gather/include/ops_mpi.hpp"

#include <boost/mpi/communicator.hpp>
#include <boost/mpi/environment.hpp>
#include <boost/serialization/vector.hpp>
#include <cmath>
#include <iostream>
#include <vector>

namespace markin_i_gather {

MyGatherMpiTask::MyGatherMpiTask(ppc::core::TaskDataPtr task_data, boost::mpi::communicator world)
    : Task(task_data), world_(world), world_rank_(world_.rank()), world_size_(world_.size()), root_(0) {}

bool MyGatherMpiTask::ValidationImpl() { return true; }

bool MyGatherMpiTask::PreProcessingImpl() {
  int_send_data_ = world_rank_;
  float_send_data_ = static_cast<float>(world_rank_) * 1.1f;
  double_send_data_ = static_cast<double>(world_rank_) * 2.2;
  return true;
}

bool MyGatherMpiTask::RunImpl() {
  MyGather(int_send_data_, 1, root_, int_recv_data_);
  MyGather(float_send_data_, 1, root_, float_recv_data_);
  MyGather(double_send_data_, 1, root_, double_recv_data_);
  return true;
}

bool MyGatherMpiTask::PostProcessingImpl() { return true; }

int calculate_tree_depth(int num_processes) {
  return static_cast<int>(std::ceil(std::log2(static_cast<double>(num_processes))));
}

template <typename T>
int MyGatherMpiTask::MyGather(const T& sendbuf, int sendcount, int root, std::vector<T>& recvbuf) {
  if (world_rank_ == root) {
    recvbuf.resize(world_size_);
    recvbuf[root] = sendbuf;

    for (int i = 0; i < world_size_; ++i) {
      if (i != root) {
        world_.recv(i, 0, recvbuf[i]);
      }
    }
  } else {
    world_.send(root, 0, sendbuf);
  }
  return 0;
}

}  // namespace markin_i_gather