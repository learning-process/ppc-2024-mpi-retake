#pragma once

#include <boost/mpi/collectives.hpp>
#include <boost/mpi/communicator.hpp>
#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

namespace makadrai_a_sobel_mpi {

class Sobel : public ppc::core::Task {
 public:
  explicit Sobel(ppc::core::TaskDataPtr task_data) : Task(std::move(task_data)) {}
  bool PreProcessingImpl() override;
  bool ValidationImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

 private:
  size_t height_img;
  size_t width_img;
  size_t peding = 2;

  std::vector<size_t> img;
  std::vector<size_t> simg;
  boost::mpi::communicator world;
};

}  // namespace makadrai_a_sobel_mpi