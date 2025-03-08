#pragma once

#include <boost/mpi/collectives.hpp>
#include <boost/mpi/communicator.hpp>
#include <cstdint>
#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

namespace konkov_i_linear_hist_stretch_mpi {

class LinearHistStretchMPI : public ppc::core::Task {
 public:
  explicit LinearHistStretchMPI(ppc::core::TaskDataPtr task_data) : Task(std::move(task_data)) {}
  bool PreProcessingImpl() override;
  bool ValidationImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

 private:
  std::vector<uint8_t> input_, output_;
  uint8_t min_intensity_, max_intensity_;
  boost::mpi::communicator world_;

  std::pair<uint8_t, uint8_t> ComputeLocalMinMax();
  void ApplyLinearStretch();
};

}  // namespace konkov_i_linear_hist_stretch_mpi
