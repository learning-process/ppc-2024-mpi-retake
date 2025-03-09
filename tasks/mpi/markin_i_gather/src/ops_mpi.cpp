#include "mpi/markin_i_gather/include/ops_mpi.hpp"

#include <boost/mpi/communicator.hpp>
#include <cmath>
#include <vector>

#include "core/task/include/task.hpp"

namespace markin_i_gather {

MyGatherMpiTask::MyGatherMpiTask(ppc::core::TaskDataPtr task_data, boost::mpi::communicator world)
    : Task(std::move(task_data)),
      world_(std::move(world)),
      world_rank_(world_.rank()),
      world_size_(world_.size()),
      root_(0) {}

bool MyGatherMpiTask::ValidationImpl() { return true; }

bool MyGatherMpiTask::PreProcessingImpl() {
  int_send_data_ = world_rank_;
  float_send_data_ = static_cast<float>(world_rank_) * 1.1F;
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

namespace {

int CalculateTreeDepth(int num_processes) {
  return static_cast<int>(std::ceil(std::log2(static_cast<double>(num_processes))));
}

}  // namespace
}  // namespace markin_i_gather