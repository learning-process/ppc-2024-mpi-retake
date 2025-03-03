#include "mpi/sharamygina_i_horizontal_line_filtraiton/include/ops_mpi.h"

#include <mpi.h>

#include <boost/mpi/collectives.hpp>
#include <boost/mpi/collectives/broadcast.hpp>
#include <boost/mpi/communicator.hpp>
#include <cstdlib>
#include <vector>

bool sharamygina_i_horizontal_line_filtration_mpi::HorizontalLineFiltrationMpi::PreProcessingImpl() {
  if (world_.rank() == 0) {
    rows_ = task_data->inputs_count[0];
    cols_ = task_data->inputs_count[1];

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

  MPI_Status status;
  int count_of_proc = world_.size();
  int myrank = world_.rank();
  result_data_.resize(rows_ * cols_);
  unsigned int block_on_proc = rows_ / count_of_proc;
  unsigned int remainder = rows_ % count_of_proc;
  std::vector<unsigned int> temporary_image((block_on_proc * cols_) + (2 * cols_));
  std::vector<unsigned int> local_data(block_on_proc * cols_);

  if (myrank == 0) {
    if (count_of_proc != 1) {
      temporary_image.resize((block_on_proc + remainder + 1) * cols_);
      for (unsigned int i = 0; i < (block_on_proc + remainder + 1) * cols_; i++) {
        temporary_image[i] = original_data_[i];
      }
    }
    for (unsigned int i = 1; i < count_of_proc; i++) {
      unsigned int count = (block_on_proc * cols_) + cols_;
      if (i != count_of_proc - 1) {
        count += cols_;
      }
      MPI_Send(original_data_.data() + ((i * block_on_proc * cols_) + ((remainder - 1) * cols_)), count, MPI_UNSIGNED,
               i, 0, MPI_COMM_WORLD);
    }
  } else {
    if (myrank != count_of_proc - 1) {
      MPI_Recv(temporary_image.data(), ((block_on_proc + 2) * cols_) + 2, MPI_UNSIGNED, 0, 0, MPI_COMM_WORLD, &status);

    } else {
      temporary_image.resize((block_on_proc + 1) * cols_);
      MPI_Recv(temporary_image.data(), (block_on_proc + 1) * cols_, MPI_UNSIGNED, 0, 0, MPI_COMM_WORLD, &status);
    }
  }

  if (myrank == count_of_proc - 1 && count_of_proc != 1) {
    for (unsigned int i = 1; i < block_on_proc; i++) {
      for (unsigned int j = 0; j < cols_; j++) {
        local_data[((i - 1) * cols_) + j] = InputAnotherPixel(temporary_image, i, j, block_on_proc + 2, cols_);
      }
    }
  } else {
    if (myrank != 0) {
      for (unsigned int i = 1; i < block_on_proc + 1; i++) {
        for (unsigned int j = 0; j < cols_; j++) {
          local_data[((i - 1) * cols_) + j] = InputAnotherPixel(temporary_image, i, j, block_on_proc + 2, cols_);
        }
      }
    } else {
      if (count_of_proc != 1) {
        for (unsigned int i = 0; i < block_on_proc + remainder; i++) {
          for (unsigned int j = 0; j < cols_; j++) {
            result_data_[(i * cols_) + j] = InputAnotherPixel(temporary_image, i, j, block_on_proc + 2, cols_);
          }
        }
      } else {
        for (unsigned int i = 0; i < rows_; i++) {
          for (unsigned int j = 0; j < cols_; j++) {
            result_data_[(i * cols_) + j] = InputAnotherPixel(original_data_, i, j, rows_, cols_);
          }
        }
      }
    }
  }

  if (myrank != 0) {
    MPI_Send(local_data.data(), block_on_proc * cols_, MPI_UNSIGNED, 0, 0, MPI_COMM_WORLD);

  } else {
    for (int i = 1; i < count_of_proc; i++) {
      MPI_Recv(result_data_.data() + ((block_on_proc + remainder) * cols_) + ((i - 1) * block_on_proc * cols_),
               block_on_proc * cols_, MPI_UNSIGNED, i, 0, MPI_COMM_WORLD, &status);
    }
  }
  return true;
}

bool sharamygina_i_horizontal_line_filtration_mpi::HorizontalLineFiltrationMpi::PostProcessingImpl() {
  if (world_.rank() == 0) {
    auto* output_ptr = reinterpret_cast<unsigned int*>(task_data->outputs[0]);
    std::ranges::copy(result_data_, output_ptr);
  }
  return true;
}

unsigned int sharamygina_i_horizontal_line_filtration_mpi::HorizontalLineFiltrationMpi::InputAnotherPixel(
    const std::vector<unsigned int>& image, unsigned int x, unsigned int y, unsigned int rows, unsigned int cols) {
  if (x < 1 || x >= rows - 1 || y < 1 || y >= cols - 1) {
    return 0;
  }
  unsigned int sum = 0;
  for (unsigned int i = 0; i < 3; i++) {
    for (unsigned int j = 0; j < 3; j++) {
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
