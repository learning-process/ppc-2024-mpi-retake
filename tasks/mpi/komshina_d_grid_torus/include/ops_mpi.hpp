#pragma once

#include <boost/mpi/collectives.hpp>
#include <boost/mpi/communicator.hpp>
#include <string>
#include <utility>
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
  
  struct TaskData {
    std::vector<int> path;
    std::vector<char> payload;
    int target;

    TaskData() = default;
    TaskData(const std::string& str, int trg) : payload(str.begin(), str.end()), target(trg) {}
  };

  static std::vector<int> CalculateRoute(int dest, int sizeX, int sizeY);

 private:
  boost::mpi::communicator world_;
  TaskData task_data_;
  int sizeX, sizeY;
  int rankX, rankY;
  int left, right, up, down;
};

}  // namespace komshina_d_grid_torus_mpi