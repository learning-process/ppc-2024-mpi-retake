#pragma once

#include <boost/mpi/collectives.hpp>
#include <boost/mpi/communicator.hpp>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <unordered_map>
#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

namespace leontev_n_binary_mpi {

class BinarySegmentsMPI : public ppc::core::Task {
 public:
  explicit BinarySegmentsMPI(ppc::core::TaskDataPtr task_data) : Task(std::move(task_data)) {}
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

 private:
  size_t GetIndex(size_t i, size_t j) const;
  boost::mpi::communicator world_;
  std::vector<uint8_t> input_image_;
  std::vector<uint8_t> local_image_;
  std::vector<uint32_t> labels_;
  size_t rows_;
  size_t cols_;
};
}  // namespace leontev_n_binary_mpi
