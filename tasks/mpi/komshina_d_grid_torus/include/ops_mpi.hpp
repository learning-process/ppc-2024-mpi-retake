#pragma once

#include <boost/mpi/collectives.hpp>
#include <boost/mpi/communicator.hpp>
#include <string>
#include <memory>

#include "core/task/include/task.hpp"

namespace komshina_d_grid_torus_mpi {
class TestTaskMPI : public ppc::core::Task {
 public:
  struct InputData {
    std::vector<int> path;
    std::vector<char> payload;
    int target;

    InputData() = default;
    InputData(const std::string& str, int dest) : payload(str.begin(), str.end()), target(dest) {}

    template <typename Archive>
    void serialize(Archive& ar, unsigned int) {
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

  static std::vector<int> CalculateRoute(int dest, int size_x, int size_y);

 private:
  boost::mpi::communicator world_;
  InputData input_data_;
  int size_x{}, size_y{};
};

}  // namespace komshina_d_grid_torus_mpi