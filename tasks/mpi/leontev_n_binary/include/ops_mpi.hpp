#pragma once

#include <boost/mpi/collectives.hpp>
#include <boost/mpi/communicator.hpp>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <set>
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
  [[nodiscard]] size_t GetIndex(size_t i, size_t j) const;
  void RootLoop(std::vector<int>& offsets);
  void RootLoopProcess(size_t border, size_t col, std::vector<std::set<uint32_t>>& label_equivalences);
  void LocalLoop(size_t local_size, uint32_t& next_label, std::vector<uint32_t>& local_labels,
                 std::vector<std::set<uint32_t>>& local_label_equivalences);
  void LocalLoopProcess(size_t row, size_t col, uint32_t& next_label, std::vector<uint32_t>& local_labels,
                        std::vector<std::set<uint32_t>>& local_label_equivalences);
  boost::mpi::communicator world_;
  std::vector<uint8_t> input_image_;
  std::vector<uint8_t> local_image_;
  std::vector<uint32_t> labels_;
  size_t rows_;
  size_t cols_;
};
}  // namespace leontev_n_binary_mpi
