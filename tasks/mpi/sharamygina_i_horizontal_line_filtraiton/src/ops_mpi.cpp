#include "mpi/sharamygina_i_horizontal_line_filtraiton/include/ops_mpi.h"

#include <mpi.h>

#include <algorithm>
#include <boost/mpi/collectives.hpp>
#include <boost/mpi/collectives/broadcast.hpp>
#include <boost/mpi/communicator.hpp>
#include <cstdlib>
#include <vector>

bool sharamygina_i_horizontal_line_filtration_mpi::HorizontalLineFiltrationMpi::PreProcessingImpl() {
  if (world_.rank() == 0) {
    rows_ = static_cast<int>(task_data->inputs_count[0]);
    cols_ = static_cast<int>(task_data->inputs_count[1]);

    auto* input_buffer = reinterpret_cast<unsigned int*>(task_data->inputs[0]);
    original_data_.assign(input_buffer, input_buffer + (rows_ * cols_));
    result_data_.resize(rows_ * cols_, 0);
  }

  return true;
}

bool sharamygina_i_horizontal_line_filtration_mpi::HorizontalLineFiltrationMpi::ValidationImpl() {
  if (world_.rank() == 0) {
    if (!task_data || task_data->inputs.empty() || task_data->inputs_count.size() < 2 || task_data->outputs.empty() ||
        task_data->outputs_count.empty() || task_data->inputs_count[0] < 3 || task_data->inputs_count[1] < 3 ||
        task_data->inputs_count[0] < static_cast<unsigned int>(world_.size())) {
      return false;
    }
  }
  return true;
}

bool sharamygina_i_horizontal_line_filtration_mpi::HorizontalLineFiltrationMpi::RunImpl() {
  boost::mpi::broadcast(world_, rows_, 0);
  boost::mpi::broadcast(world_, cols_, 0);

  int count_of_proc = world_.size();
  int myrank = world_.rank();

  result_data_.resize(rows_ * cols_);
  std::vector<unsigned int> temporary_image;
  std::vector<unsigned int> local_data;

  PrepareTemporaryImage(myrank, count_of_proc, temporary_image);
  ReceiveData(myrank, count_of_proc, temporary_image);
  ProcessLocalData(myrank, count_of_proc, temporary_image, local_data);
  SendData(myrank, count_of_proc, local_data);

  return true;
}

void sharamygina_i_horizontal_line_filtration_mpi::HorizontalLineFiltrationMpi::PrepareTemporaryImage(
    int myrank, int count_of_proc, std::vector<unsigned int>& temporary_image) {
  int block_on_proc = rows_ / count_of_proc;
  int remainder = rows_ % count_of_proc;

  if (myrank == 0 && count_of_proc != 1) {
    temporary_image.resize((block_on_proc + remainder + 1) * cols_);
    for (int i = 0; i < (block_on_proc + remainder + 1) * cols_; i++) {
      temporary_image[i] = original_data_[i];
    }
  } else {
    temporary_image.resize((block_on_proc + 2) * cols_);
  }
}

void sharamygina_i_horizontal_line_filtration_mpi::HorizontalLineFiltrationMpi::ReceiveData(
    int myrank, int count_of_proc, std::vector<unsigned int>& temporary_image) {
  MPI_Status status;
  int block_on_proc = rows_ / count_of_proc;
  int remainder = rows_ % count_of_proc;

  if (myrank == 0) {
    for (int i = 1; i < count_of_proc; i++) {
      int count = (block_on_proc * cols_) + cols_;
      if (i != count_of_proc - 1) {
        count += cols_;
      }
      MPI_Send(original_data_.data() + ((i * block_on_proc * cols_) + ((remainder - 1) * cols_)), count, MPI_UNSIGNED,
               i, 0, MPI_COMM_WORLD);
    }
  } else {
    int count = (myrank != count_of_proc - 1) ? ((block_on_proc + 2) * cols_) : ((block_on_proc + 1) * cols_);
    MPI_Recv(temporary_image.data(), count, MPI_UNSIGNED, 0, 0, MPI_COMM_WORLD, &status);
  }
}

void sharamygina_i_horizontal_line_filtration_mpi::HorizontalLineFiltrationMpi::ProcessLocalData(
    int myrank, int count_of_proc, const std::vector<unsigned int>& temporary_image,
    std::vector<unsigned int>& local_data) {
  int block_on_proc = rows_ / count_of_proc;
  int remainder = rows_ % count_of_proc;
  local_data.resize(block_on_proc * cols_);

  if (myrank == count_of_proc - 1 && count_of_proc != 1) {
    ProcessLastRank(myrank, block_on_proc, temporary_image, local_data);
  } else if (myrank == 0) {
    ProcessFirstRank(count_of_proc, block_on_proc, remainder, temporary_image);
  } else {
    ProcessMiddleRanks(myrank, block_on_proc, temporary_image, local_data);
  }
}

void sharamygina_i_horizontal_line_filtration_mpi::HorizontalLineFiltrationMpi::ProcessLastRank(
    int myrank, int block_on_proc, const std::vector<unsigned int>& temporary_image,
    std::vector<unsigned int>& local_data) {
  for (int i = 1; i < block_on_proc; i++) {
    for (int j = 0; j < cols_; j++) {
      local_data[((i - 1) * cols_) + j] = InputAnotherPixel(temporary_image, i, j, block_on_proc + 2, cols_);
    }
  }
}

void sharamygina_i_horizontal_line_filtration_mpi::HorizontalLineFiltrationMpi::ProcessFirstRank(
    int count_of_proc, int block_on_proc, int remainder, const std::vector<unsigned int>& temporary_image) {
  if (count_of_proc != 1) {
    for (int i = 0; i < block_on_proc + remainder; i++) {
      for (int j = 0; j < cols_; j++) {
        result_data_[(i * cols_) + j] = InputAnotherPixel(temporary_image, i, j, block_on_proc + 2, cols_);
      }
    }
  } else {
    for (int i = 0; i < rows_; i++) {
      for (int j = 0; j < cols_; j++) {
        result_data_[(i * cols_) + j] = InputAnotherPixel(original_data_, i, j, rows_, cols_);
      }
    }
  }
}

void sharamygina_i_horizontal_line_filtration_mpi::HorizontalLineFiltrationMpi::ProcessMiddleRanks(
    int myrank, int block_on_proc, const std::vector<unsigned int>& temporary_image,
    std::vector<unsigned int>& local_data) {
  for (int i = 1; i < block_on_proc + 1; i++) {
    for (int j = 0; j < cols_; j++) {
      local_data[((i - 1) * cols_) + j] = InputAnotherPixel(temporary_image, i, j, block_on_proc + 2, cols_);
    }
  }
}

void sharamygina_i_horizontal_line_filtration_mpi::HorizontalLineFiltrationMpi::SendData(
    int myrank, int count_of_proc, const std::vector<unsigned int>& local_data) {
  MPI_Status status;
  int block_on_proc = rows_ / count_of_proc;
  int remainder = rows_ % count_of_proc;

  if (myrank != 0) {
    MPI_Send(local_data.data(), block_on_proc * cols_, MPI_UNSIGNED, 0, 0, MPI_COMM_WORLD);
  } else {
    for (int i = 1; i < count_of_proc; i++) {
      MPI_Recv(result_data_.data() + ((block_on_proc + remainder) * cols_) + ((i - 1) * block_on_proc * cols_),
               block_on_proc * cols_, MPI_UNSIGNED, i, 0, MPI_COMM_WORLD, &status);
    }
  }
}

bool sharamygina_i_horizontal_line_filtration_mpi::HorizontalLineFiltrationMpi::PostProcessingImpl() {
  if (world_.rank() == 0) {
    auto* output_ptr = reinterpret_cast<unsigned int*>(task_data->outputs[0]);
    std::ranges::copy(result_data_, output_ptr);
  }
  return true;
}

unsigned int sharamygina_i_horizontal_line_filtration_mpi::HorizontalLineFiltrationMpi::InputAnotherPixel(
    const std::vector<unsigned int>& image, int x, int y, int rows, int cols) {
  if (x < 1 || x >= rows - 1 || y < 1 || y >= cols - 1) {
    return 0;
  }
  unsigned int sum = 0;
  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 3; j++) {
      int t_x = x + i - 1;
      int t_y = y + j - 1;
      if (t_x < 0 || t_x > rows - 1) {
        t_x = x;
      }
      if (t_y < 0 || t_y > cols - 1) {
        t_y = y;
      }
      if ((t_x * cols) + t_y >= cols * rows) {
        t_x = x;
        t_y = y;
      }
      sum += static_cast<unsigned int>(image[(t_x * cols) + t_y] * (gauss_[i][j]));
    }
  }
  return sum / 16;
}
