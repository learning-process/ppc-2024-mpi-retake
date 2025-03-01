#pragma once
#include <boost/mpi/collectives.hpp>
#include <boost/mpi/communicator.hpp>
#include <memory>
#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

namespace sharamygina_i_vector_dot_product_mpi {
class VectorDotProductMpi : public ppc::core::Task {
 public:
  explicit VectorDotProductMpi(std::shared_ptr<ppc::core::TaskData> task_data) : Task(std::move(task_data)) {}
  bool PreProcessingImpl() override;
  bool ValidationImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

 private:
  std::vector<int> v1_;
  std::vector<int> v2_;
  std::vector<int> local_v1_;
  std::vector<int> local_v2_;
  int res_{};
  boost::mpi::communicator world_;
  unsigned int delta_;
};
}  // namespace sharamygina_i_vector_dot_product_mpi