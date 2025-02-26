#include <cstddef>
#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

namespace malyshev_v_lent_horizontal_seq {

std::vector<double> GetRandomMatrix(size_t rows, size_t cols);
std::vector<double> GetRandomVector(size_t size);

class MatrixVectorMultiplication : public ppc::core::Task {
 public:
  explicit MatrixVectorMultiplication(ppc::core::TaskDataPtr task_data) : Task(std::move(task_data)) {}
  bool PreProcessingImpl() override;
  bool ValidationImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

 private:
  std::vector<double> matrix_;
  std::vector<double> vector_;
  std::vector<double> result_;
  size_t rows_, cols_;
};

}  // namespace malyshev_v_lent_horizontal_seq