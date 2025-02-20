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
int GridTorus(int sourceRank, int targetRank, int gridSizeX, int gridSizeY,
                                         bool isHorizontalClosed, bool isVerticalClosed);
class TestTaskMPI : public ppc::core::Task {
 public:
  explicit TestTaskMPI(ppc::core::TaskDataPtr task_data) : Task(std::move(task_data)) {}
  bool PreProcessingImpl() override;
  bool ValidationImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

  struct RoutingPacket {
    int message_payload;
    bool is_routing_complete;
    int target_rank;
    std::vector<int> routing_path;

    template <class Archive>
    void serialize(Archive& ar, unsigned int version) {
      ar & message_payload;
      ar & is_routing_complete;
      ar & target_rank;
      ar & routing_path;
    }
  };

 private:
  RoutingPacket routing_packet;
  int grid_columns;
  int grid_rows;
  boost::mpi::communicator world;
};

}  // namespace komshina_d_grid_torus_mpi