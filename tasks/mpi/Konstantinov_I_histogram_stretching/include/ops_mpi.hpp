#pragma once

#include <boost/mpi/collectives.hpp>
#include <boost/mpi/communicator.hpp>
#include <memory>
#include <numeric>
#include <string>
#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

namespace konstantinov_i_linear_histogram_stretch_mpi {

class LinearHistogramStretchSeq : public ppc::core::Task {
 public:
  explicit LinearHistogramStretchSeq(std::shared_ptr<ppc::core::TaskData> task_data) : Task(std::move(task_data)) {}
  bool PreProcessingImpl() override;
  bool ValidationImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

 private:
  std::vector<int> I;
  std::vector<int> image_input;
  std::vector<int> image_output;
};

class LinearHistogramStretchMpi : public ppc::core::Task {
 public:
  explicit LinearHistogramStretchMpi(std::shared_ptr<ppc::core::TaskData> task_data) : Task(std::move(task_data)) {}
  bool PreProcessingImpl() override;
  bool ValidationImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

 private:
  std::vector<int> image_input;
  std::vector<int> image_output;
  boost::mpi::communicator world;
};
}  // namespace konstantinov_i_linear_histogram_stretch_mpi