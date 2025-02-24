#pragma once

#include <boost/mpi/collectives.hpp>
#include <boost/mpi/communicator.hpp>
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
  
  static std::vector<int> calculate_route(int dest, int world_size, int width);

  struct InputData {
    std::vector<int> path;
    std::vector<char> payload;
    int target;

    InputData() = default;
    InputData(const std::string& str, int target_node) : payload(str.begin(), str.end()), target(target_node) {}

    template <class Archive>
    void serialize(Archive& ar, unsigned int version) {
      ar & target;
      ar & payload;
      ar & path;
    }
  };

 private:
  InputData input_data_;
  boost::mpi::communicator world_;
};

}  // namespace komshina_d_grid_torus_mpi