#pragma once

#include <gtest/gtest.h>

#include <boost/mpi/collectives.hpp>
#include <boost/mpi/communicator.hpp>
#include <cmath>
#include <memory>
#include <utility>
#include <vector>

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
  
  static std::vector<int> calculate_route(int dest, int sizeX, int sizeY);

 private:
  boost::mpi::communicator world;
  InputData inputData;
  int sizeX{}, sizeY{};
};

}  // namespace komshina_d_grid_torus_mpi