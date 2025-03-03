#pragma once

#include <boost/mpi/collectives.hpp>
#include <boost/mpi/communicator.hpp>
#include <memory>
#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

namespace konstantinov_i_gauss_jordan_method_mpi {

std::vector<double> ProcessMatrix(int n, int k, const std::vector<double>& matrix);
void CalcSizesDispls(int n, int k, int world_size, std::vector<int>& sizes, std::vector<int>& displs);
std::vector<std::pair<int, int>> GetIndicies(int rows, int cols);
void UpdateMatrix(int n, int k, std::vector<double>& matrix, const std::vector<double>& iter_result);
bool IsNonSingularSystem(const std::vector<double>& a, int n);
void GatherResults(boost::mpi::communicator& world, std::vector<double>& local_result, std::vector<double>& iter_result,
                   std::vector<int>& sizes, std::vector<int>& displs);
std::vector<double> CalculateLocalResult(int local_size, std::vector<std::pair<int, int>>& local_indicies,
                                         const std::vector<double>& iter_matrix, int n, int k);
std::vector<std::pair<int, int>> ScatterIndices(boost::mpi::communicator& world,
                                                std::vector<std::pair<int, int>>& indicies, std::vector<int>& sizes,
                                                std::vector<int>& displs, int local_size);
void BroadcastData(boost::mpi::communicator& world, std::vector<int>& sizes, std::vector<double>& iter_matrix);
bool BroadcastSolve(boost::mpi::communicator& world, bool solve);
void PrepareIteration(int n, int k, const std::vector<double>& matrix, std::vector<double>& iter_matrix,
                      std::vector<int>& sizes, std::vector<int>& displs, std::vector<std::pair<int, int>>& indicies,
                      std::vector<double>& iter_result, boost::mpi::communicator& world);
void SwapRows(int n, int k, int change, std::vector<double>& matrix);
bool SwapRowsIfZero(int n, int k, std::vector<double>& matrix, bool& solve);


class GaussJordanMethodMPI : public ppc::core::Task {
 public:
  explicit GaussJordanMethodMPI(std::shared_ptr<ppc::core::TaskData> task_data) : Task(std::move(task_data)) {}
  bool PreProcessingImpl() override;
  bool ValidationImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

 private:
  std::vector<double> matrix;
  bool solve = true;
  int n;
  std::vector<int> sizes;
  std::vector<int> displs;
  std::vector<double> iter_matrix;
  std::vector<double> iter_result;
  std::vector<std::pair<int, int>> indicies;
  boost::mpi::communicator world;
};

class GaussJordanMethodSeq : public ppc::core::Task {
 public:
  explicit GaussJordanMethodSeq(std::shared_ptr<ppc::core::TaskData> task_data) : Task(std::move(task_data)) {}
  bool PreProcessingImpl() override;
  bool ValidationImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

 private:
  std::vector<double> matrix;
  bool solve = true;
  int n;
};

}  // namespace konstantinov_i_gauss_jordan_method_mpi