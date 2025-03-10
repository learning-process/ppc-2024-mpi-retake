
#pragma once

#include <boost/mpi/collectives.hpp>
#include <boost/mpi/communicator.hpp>
#include <boost/serialization/vector.hpp>  //NOLINT
#include <cstdint>
#include <limits>
#include <utility>
#include <vector>

#include "core/task/include/task.hpp"
namespace vinyaikina_e_max_of_vector_elements {

class VectorMaxSeq : public ppc::core::Task {
 public:
  explicit VectorMaxSeq(ppc::core::TaskDataPtr task_data) : Task(std::move(task_data)) {}
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
  explicit VectorMaxPar(ppc::core::TaskDataPtr task_data) : Task(std::move(task_data)) {}
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
