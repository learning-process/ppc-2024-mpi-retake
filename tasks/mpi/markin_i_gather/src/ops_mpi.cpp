#include "mpi/markin_i_gather/include/ops_mpi.hpp"
#include <iostream>
#include <vector>
#include <boost/mpi/environment.hpp>
#include <boost/mpi/communicator.hpp>
#include <boost/serialization/vector.hpp>

namespace markin_i_gather {

MyGatherMpiTask::MyGatherMpiTask(ppc::core::TaskDataPtr task_data, boost::mpi::communicator world)
    : Task(task_data),
      world_(world),
      world_rank_(world_.rank()),
      world_size_(world_.size()),
      root_(0) {}

bool MyGatherMpiTask::ValidationImpl() {
    return true; 
}

bool MyGatherMpiTask::PreProcessingImpl() {
    int_send_data_ = world_rank_;
    float_send_data_ = (float)world_rank_ * 1.1f;
    double_send_data_ = (double)world_rank_ * 2.2;
    return true;
}

bool MyGatherMpiTask::RunImpl() {
    My_Gather(int_send_data_, 1, root_, int_recv_data_);
    My_Gather(float_send_data_, 1, root_, float_recv_data_);
    My_Gather(double_send_data_, 1, root_, double_recv_data_);
    return true;
}

bool MyGatherMpiTask::PostProcessingImpl() {
    return true;
}
int calculate_tree_depth(int num_processes) {
    int depth = 0;
    int nodes = 1;
    while (nodes < num_processes) {
        nodes *= 2;
        depth++;
    }
    return depth;
}

template <typename T>
int MyGatherMpiTask::My_Gather(const T& sendbuf, int sendcount, int root, std::vector<T>& recvbuf) {

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