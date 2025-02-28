
#pragma once

#include <boost/mpi/communicator.hpp>
#include <cstdint>
#include <limits>
#include <memory>
#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

namespace vinyaikina_e_max_of_vector_elements {

[[nodiscard]] std::vector<int32_t> MakeRandomVector(int32_t size, int32_t val_min, int32_t val_max);

class VectorMaxSeq : public ppc::core::Task {
 public:
  explicit VectorMaxSeq(std::shared_ptr<ppc::core::TaskData> task_data) : Task(std::move(task_data)) {}
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

 private:
  std::vector<int32_t> input_;
  int32_t max_ = std::numeric_limits<int32_t>::min();
};

class VectorMaxPar : public ppc::core::Task {
 public:
  explicit VectorMaxPar(std::shared_ptr<ppc::core::TaskData> task_data) : Task(std::move(task_data)) {}
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

 private:
  std::vector<int32_t> input_, local_input_;
  int32_t max_ = std::numeric_limits<int32_t>::min();
  boost::mpi::communicator world_;
};

}  // namespace vinyaikina_e_max_of_vector_elements
