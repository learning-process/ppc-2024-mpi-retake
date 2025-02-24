#pragma once

#include <utility>
#include <string>
#include <boost/mpi/collectives.hpp>
#include <boost/mpi/communicator.hpp>
#include <vector>

#include "core/task/include/task.hpp"

namespace komshina_d_grid_torus_mpi {

class TestTaskMPI : public ppc::core::Task {
 public:

  explicit TestTaskMPI(ppc::core::TaskDataPtr task_data) : Task(std::move(task_data)) {}
  bool PreProcessingImpl() override;
  bool ValidationImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;
  static std::vector<int> ComputePath(int target, int world_size, int width, int height);
  
  struct TaskData {
    std::vector<int> path;
    std::vector<char> payload;
    int target;

    TaskData() = default;
    TaskData(const std::string& str, int tgt) : payload(str.begin(), str.end()), target(tgt) {}

    template <class Archive>
    void serialize(Archive& ar, unsigned int version) {
      ar & target;
      ar & payload;
      ar & path;
    }
  };

 private:
  boost::mpi::communicator world_;
  TaskData task_data_;
  
  void ComputeGridSize();
  static int GetNextHop(int current, int target, int width, int height);
  int width_, height_;
};

}  // namespace komshina_d_grid_torus_mpi