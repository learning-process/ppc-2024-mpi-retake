#pragma once
#include <boost/mpi/communicator.hpp>
#include <vector>

#include "core/task/include/task.hpp"

namespace markin_i_gather {

class MyGatherMpiTask : public ppc::core::Task {
 public:
  MyGatherMpiTask(ppc::core::TaskDataPtr task_data, boost::mpi::communicator world);

  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

  void SetRoot(int root, boost::mpi::communicator& world) {
    root_ = root;
    world_ = world;
    world_rank_ = world_.rank();
    world_size_ = world_.size();
  }

  std::vector<int>& GetIntRecvData() { return int_recv_data_; }
  std::vector<float>& GetFloatRecvData() { return float_recv_data_; }
  std::vector<double>& GetDoubleRecvData() { return double_recv_data_; }

 private:
  template <typename T>
  int MyGather(const T& sendbuf, int sendcount, int root, std::vector<T>& recvbuf);

  boost::mpi::communicator world_;
  int world_rank_;
  int world_size_;
  int root_;

  std::vector<int> int_recv_data_;
  std::vector<float> float_recv_data_;
  std::vector<double> double_recv_data_;
  int int_send_data_;
  float float_send_data_;
  double double_send_data_;
};

}  // namespace markin_i_gather
