#pragma once

#include <boost/mpi/collectives.hpp>
#include <boost/mpi/communicator.hpp>
#include <memory>
#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

namespace sharamygina_i_horizontal_line_filtration_mpi {
class HorizontalLineFiltrationMpi : public ppc::core::Task {
 public:
  explicit HorizontalLineFiltrationMpi(std::shared_ptr<ppc::core::TaskData> task_data) : Task(std::move(task_data)) {}
  bool PreProcessingImpl() override;
  bool ValidationImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

 private:
  unsigned int gauss_[3][3]{{1, 2, 1}, {2, 4, 2}, {1, 2, 1}};
  std::vector<unsigned int> original_data_;
  std::vector<unsigned int> result_data_;
  int rows_;
  int cols_;
  unsigned int InputAnotherPixel(const std::vector<unsigned int>& image, int x, int y, int rows, int cols);
  void SendData(int myrank, int count_of_proc, const std::vector<unsigned int>& local_data);
  void ProcessLocalData(int myrank, int count_of_proc, const std::vector<unsigned int>& temporary_image,
                        std::vector<unsigned int>& local_data);
  void ProcessLastRank(int myrank, int block_on_proc, const std::vector<unsigned int>& temporary_image,
                       std::vector<unsigned int>& local_data);
  void ProcessFirstRank(int count_of_proc, int block_on_proc, int remainder,
                        const std::vector<unsigned int>& temporary_image);
  void ProcessMiddleRanks(int myrank, int block_on_proc, const std::vector<unsigned int>& temporary_image,
                          std::vector<unsigned int>& local_data);
  void ReceiveData(int myrank, int count_of_proc, std::vector<unsigned int>& temporary_image);
  void PrepareTemporaryImage(int myrank, int count_of_proc, std::vector<unsigned int>& temporary_image);
  boost::mpi::communicator world_;
};
}  // namespace sharamygina_i_horizontal_line_filtration_mpi