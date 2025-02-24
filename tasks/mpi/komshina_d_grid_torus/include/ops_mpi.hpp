#pragma once

#include <boost/mpi/collectives.hpp>
#include <boost/mpi/communicator.hpp>
#include <utility>
#include <string>
#include <vector>

#include "core/task/include/task.hpp"

namespace komshina_d_grid_torus_mpi {

class TestTaskMPI : public ppc::core::Task {
 public:
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

  explicit TestTaskMPI(ppc::core::TaskDataPtr task_data) : Task(std::move(task_data)) {}
  bool PreProcessingImpl() override;
  bool ValidationImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;
  [[nodiscard]] static std::vector<int> compute_path(int target, int world_size, int width, int height);
  void compute_grid_size();
  static int get_next_hop(int current, int dest, int width, int height);
  int width, height;

 private:
  boost::mpi::communicator world_;
  TaskData task_data_;

};

}  // namespace komshina_d_grid_torus_mpi