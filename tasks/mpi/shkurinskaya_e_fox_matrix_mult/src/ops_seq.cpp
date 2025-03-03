#include <algorithm>
#include <boost/mpi/collectives.hpp>
#include <boost/mpi/collectives/broadcast.hpp>
#include <cmath>
#include <random>
#include <vector>

#include "mpi/shkurinskaya_e_fox_matrix_mult/include/ops_sec.hpp"

std::vector<double> shkurinskaya_e_fox_mat_mul_mpi::GetRandomMatrix(int rows, int cols) {
  std::vector<double> result(rows * cols);

  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<> dis(-50.0, 50.0);

  for (int i = 0; i < rows; ++i) {
    for (int j = 0; j < cols; ++j) {
      result[(i * cols) + j] = dis(gen);
    }
  }

  return result;
}

bool shkurinskaya_e_fox_mat_mul_mpi::FoxMatMulMPI::PreProcessingImpl() {
  if (world_.rank() == 0) {
    matrix_size_ = (int)(task_data->inputs_count[0]);

    sz_ = 1;
    while (sz_ < matrix_size_) {
      sz_ *= 2;
    }
    sz_ = (sz_ + root_ - 1) / root_ * root_;
    block_sz_ = sz_ / root_;

    inputA_.resize(sz_ * sz_);
    inputB_.resize(sz_ * sz_);
    output_ = std::vector<double>(matrix_size_ * matrix_size_, 0.0);

    auto *it1 = reinterpret_cast<double *>(task_data->inputs[0]);
    auto *it2 = reinterpret_cast<double *>(task_data->inputs[1]);
    for (int i = 0; i < matrix_size_; ++i) {
      std::copy(it1 + (i * matrix_size_), it1 + ((i + 1) * matrix_size_), inputA_.begin() + i * sz_);
      std::copy(it2 + (i * matrix_size_), it2 + ((i + 1) * matrix_size_), inputB_.begin() + i * sz_);
    }
  }
  root_ = (int)sqrt(world_.size());
  return true;
}

bool shkurinskaya_e_fox_mat_mul_mpi::FoxMatMulMPI::ValidationImpl() {
  if (world_.rank() == 0) {
    root_ = (int)sqrt(world_.size());
    return (task_data->inputs_count[0] > 0) && (task_data->inputs_count[0] == task_data->outputs_count[0]) &&
           (root_ * root_ == world_.size());
  }
  return true;
}

static void ShareData(boost::mpi::communicator &world_, int block_sz_, int sz_, int root_,
                      std::vector<double> &left_block, std::vector<double> &right_block, std::vector<double> &inputA_,
                      std::vector<double> &inputB_) {
  if (world_.rank() == 0) {
    int block_matix_size = block_sz_ * block_sz_;
    std::vector<double> left_to_send(block_matix_size, 0.0);
    std::vector<double> right_to_send(block_matix_size, 0.0);

    for (int i = 0; i < block_sz_; ++i) {
      for (int j = 0; j < block_sz_; ++j) {
        left_block[(i * block_sz_) + j] = inputA_[(i * sz_) + j];
        right_block[(i * block_sz_) + j] = inputB_[(i * sz_) + j];
      }
    }

    for (int proc = 1; proc < world_.size(); ++proc) {
      int local_color = proc / root_;
      int local_key = proc % root_;

      for (int i = 0; i < block_sz_; i++) {
        for (int j = 0; j < block_sz_; j++) {
          left_to_send[(i * block_sz_) + j] =
              inputA_[((local_color * block_sz_ + i) * sz_) + (local_key * block_sz_) + j];
          right_to_send[(i * block_sz_) + j] =
              inputB_[((local_color * block_sz_ + i) * sz_) + (local_key * block_sz_) + j];
        }
      }

      world_.send(proc, 0, left_to_send);
      world_.send(proc, 0, right_to_send);
    }
  } else {
    world_.recv(0, 0, left_block);
    world_.recv(0, 0, right_block);
  }
}

static void SaveMatrix(std::vector<double> &left_block, std::vector<double> &right_block, std::vector<double> &out,
                       int block_sz_, int sz_) {
  for (int i = 0; i < block_sz_; ++i) {
    for (int j = 0; j < block_sz_; ++j) {
      for (int k = 0; k < block_sz_; ++k) {
        out[(i * sz_) + j] += left_block[(i * block_sz_) + k] * right_block[(k * block_sz_) + j];
      }
    }
  }
}

static void GatherResult(boost::mpi::communicator world_, int root_, int block_sz_, int sz_, int matrix_size_,
                         std::vector<double> &local_res, std::vector<double> &temp_output,
                         std::vector<double> &output_) {
  // Gather results on rank 0
  if (world_.rank() == 0) {
    for (int proc = 1; proc < world_.size(); proc++) {
      world_.recv(proc, 0, local_res);
      int local_color = proc / root_;
      int local_key = proc % root_;
      int mergin_x = block_sz_ * local_color;
      int mergin_y = block_sz_ * local_key;

      for (int i = 0; i < block_sz_; i++) {
        for (int j = 0; j < block_sz_; j++) {
          temp_output[((mergin_x + i) * sz_) + j + mergin_y] = local_res[(i * block_sz_) + j];
        }
      }
    }
  } else {
    world_.send(0, 0, local_res);
  }

  if (world_.rank() == 0) {
    for (int i = 0; i < matrix_size_; ++i) {
      for (int j = 0; j < matrix_size_; ++j) {
        output_[(i * matrix_size_) + j] = temp_output[(i * sz_) + j];
      }
    }
  }
}

bool shkurinskaya_e_fox_mat_mul_mpi::FoxMatMulMPI::RunImpl() {
  // share size and block size
  boost::mpi::broadcast(world_, block_sz_, 0);
  boost::mpi::broadcast(world_, sz_, 0);

  int block_matix_size = block_sz_ * block_sz_;
  std::vector<double> local_res(block_matix_size, 0.0);
  std::vector<double> left_block(block_matix_size, 0.0);
  std::vector<double> right_block(block_matix_size, 0.0);

  // share blocks
  ShareData(world_, block_sz_, sz_, root_, left_block, right_block, inputA_, inputB_);
  std::vector<double> recv_block_result(block_sz_ * block_sz_, 0.0);

  int color = world_.rank() / root_;
  int key = world_.rank() % root_;
  // create row communicator to broadcast A~i~(i + k)
  boost::mpi::communicator row_comm = world_.split(color, key);

  std::vector<double> temp = left_block;
  std::vector<double> temp_output(sz_ * sz_, 0.0);

  bool valid = temp.empty();

  // Fox's algorithm (root_ steps)
  for (int it = 0; it < root_; ++it) {
    // (color + key + it) % root_  equals A~i~(i + k)
    if ((((color + key + it) % root_) == 0) && (!valid)) {
      valid = true;
      left_block = std::move(temp);
    }

    // Broadcast the local block to all ranks in the row
    boost::mpi::broadcast(row_comm, left_block.data(), (int)left_block.size(), (root_ - color - it + root_) % root_);

    // Matrix multiplication
    if (world_.rank() == 0) {
      SaveMatrix(left_block, right_block, temp_output, block_sz_, sz_);
    } else {
      SaveMatrix(left_block, right_block, local_res, block_sz_, block_sz_);
    }
    // do not need to send/recv right block
    if (it == root_ - 1) {
      break;
    }

    int prev = (((world_.rank() / root_ - 1 + root_) % root_) * root_) + key;
    int next = (((world_.rank() / root_ + 1) % root_) * root_) + key;

    if (color == 0) {
      std::vector<double> tmp(right_block.size());
      world_.recv(prev, 0, tmp);
      world_.send(next, 0, right_block);
      right_block = tmp;
    } else {
      world_.send(next, 0, right_block);
      world_.recv(prev, 0, right_block);
    }
  }
  GatherResult(world_, root_, block_sz_, sz_, matrix_size_, local_res, temp_output, output_);
  return true;
}

bool shkurinskaya_e_fox_mat_mul_mpi::FoxMatMulMPI::PostProcessingImpl() {
  if (world_.rank() == 0) {
    auto *it1 = reinterpret_cast<double *>(task_data->outputs[0]);
    std::ranges::copy(output_, it1);
  }
  return true;
}
