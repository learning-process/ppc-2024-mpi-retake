#include <algorithm>
#include <boost/mpi/collectives.hpp>
#include <boost/mpi/collectives/broadcast.hpp>
#include <cmath>
#include <random>
#include <vector>

#include "mpi/shkurinskaya_e_fox_matrix_mult/include/ops_sec.hpp"

namespace shkurinskaya_e_fox_mat_mul_mpi {
    std::vector<double> GetRandomMatrix(int rows, int cols);
}

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
namespace shkurinskaya_e_fox_mat_mul_mpi {
void ShareData(boost::mpi::communicator &world, int block_sz, int sz, int root, std::vector<double> &left_block,
               std::vector<double> &right_block, std::vector<double> &input_a, std::vector<double> &input_b) {
  if (world.rank() == 0) {
    int block_matix_size = block_sz * block_sz;
    std::vector<double> left_to_send(block_matix_size, 0.0);
    std::vector<double> right_to_send(block_matix_size, 0.0);

    for (int i = 0; i < block_sz; ++i) {
      for (int j = 0; j < block_sz; ++j) {
        left_block[(i * block_sz) + j] = input_a[(i * sz) + j];
        right_block[(i * block_sz) + j] = input_b[(i * sz) + j];
      }
    }

    for (int proc = 1; proc < world.size(); ++proc) {
      int local_color = proc / root;
      int local_key = proc % root;

      for (int i = 0; i < block_sz; i++) {
        for (int j = 0; j < block_sz; j++) {
          left_to_send[(i * block_sz) + j] = input_a[((local_color * block_sz + i) * sz) + (local_key * block_sz) + j];
          right_to_send[(i * block_sz) + j] = input_b[((local_color * block_sz + i) * sz) + (local_key * block_sz) + j];
        }
      }

      world.send(proc, 0, left_to_send);
      world.send(proc, 0, right_to_send);
    }
  } else {
    world.recv(0, 0, left_block);
    world.recv(0, 0, right_block);
  }
}

void SaveMatrix(std::vector<double> &left_block, std::vector<double> &right_block, std::vector<double> &out,
                int block_sz, int sz) {
  for (int i = 0; i < block_sz; ++i) {
    for (int j = 0; j < block_sz; ++j) {
      for (int k = 0; k < block_sz; ++k) {
        out[(i * sz) + j] += left_block[(i * block_sz) + k] * right_block[(k * block_sz) + j];
      }
    }
  }
}

void GatherResult(boost::mpi::communicator &world, int root, int block_sz, int sz, int matrix_size,
                  std::vector<double> &local_res, std::vector<double> &temp_output, std::vector<double> &output) {
  // Gather results on rank 0
  if (world.rank() == 0) {
    for (int proc = 1; proc < world.size(); proc++) {
      world.recv(proc, 0, local_res);
      int local_color = proc / root;
      int local_key = proc % root;
      int mergin_x = block_sz * local_color;
      int mergin_y = block_sz * local_key;

      for (int i = 0; i < block_sz; i++) {
        for (int j = 0; j < block_sz; j++) {
          temp_output[((mergin_x + i) * sz) + j + mergin_y] = local_res[(i * block_sz) + j];
        }
      }
    }
  } else {
    world.send(0, 0, local_res);
  }

  if (world.rank() == 0) {
    for (int i = 0; i < matrix_size; ++i) {
      for (int j = 0; j < matrix_size; ++j) {
        output[(i * matrix_size) + j] = temp_output[(i * sz) + j];
      }
    }
  }
}

}  // namespace shkurinskaya_e_fox_mat_mul_mpi

bool shkurinskaya_e_fox_mat_mul_mpi::FoxMatMulMPI::RunImpl() {
  // share size and block size
  boost::mpi::broadcast(world_, block_sz_, 0);
  boost::mpi::broadcast(world_, sz_, 0);

  int block_matix_size = block_sz_ * block_sz_;
  std::vector<double> local_res(block_matix_size, 0.0);
  std::vector<double> left_block(block_matix_size, 0.0);
  std::vector<double> right_block(block_matix_size, 0.0);

  // share blocks
  shkurinskaya_e_fox_mat_mul_mpi::ShareData(world_, block_sz_, sz_, root_, left_block, right_block, inputA_, inputB_);
  std::vector<double> recv_block_result(block_sz_ * block_sz_, 0.0);

  int color = world_.rank() / root_;
  int key = world_.rank() % root_;
  // create row communicator to broadcast A~i~(i + k)
  boost::mpi::communicator row_comm = world_.split(color, key);

  std::vector<double> temp = left_block;
  std::vector<double> temp_output(sz_ * sz_, 0.0);

  // Fox's algorithm (root_ steps)
  for (int it = 0; it < root_; ++it) {
    // (color + key + it) % root_  equals A~i~(i + k)
    if (((color + key + it) % root_) == 0) {
      for (int k = 0; k < (int)temp.size(); ++k) {
        left_block[k] = temp[k];
      }
    }

    // Broadcast the local block to all ranks in the row
    boost::mpi::broadcast(row_comm, left_block.data(), (int)left_block.size(), (root_ - color - it + root_) % root_);

    // Matrix multiplication
    if (world_.rank() == 0) {
      shkurinskaya_e_fox_mat_mul_mpi::SaveMatrix(left_block, right_block, temp_output, block_sz_, sz_);
    } else {
      shkurinskaya_e_fox_mat_mul_mpi::SaveMatrix(left_block, right_block, local_res, block_sz_, block_sz_);
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
  shkurinskaya_e_fox_mat_mul_mpi::GatherResult(world_, root_, block_sz_, sz_, matrix_size_, local_res, temp_output,
                                               output_);
  return true;
}

bool shkurinskaya_e_fox_mat_mul_mpi::FoxMatMulMPI::PostProcessingImpl() {
  if (world_.rank() == 0) {
    auto *it1 = reinterpret_cast<double *>(task_data->outputs[0]);
    std::ranges::copy(output_, it1);
  }
  return true;
}
