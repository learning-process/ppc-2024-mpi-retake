#include <cmath>
#include <cstddef>
#include <random>
#include <vector>

#include "mpi/shkurinskaya_e_fox_matrix_mult/include/ops_sec.hpp"

std::vector<double> shkurinskaya_e_fox_mat_mul_mpi::getRandomMatrix(int rows, int cols) {
  std::vector<double> result(rows * cols);

  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<> dis(-50.0, 50.0);

  for (int i = 0; i < rows; ++i) {
    for (int j = 0; j < cols; ++j) {
      result[i * cols + j] = dis(gen);
    }
  }

  return result;
}

bool shkurinskaya_e_fox_mat_mul_mpi::FoxMatMulMPI::PreProcessingImpl() {
  if (world.rank() == 0) {
    matrix_size = task_data->inputs_count[0];

    sz = 1;
    while (sz < matrix_size) {
      sz *= 2;
    }
    sz = (sz + root - 1) / root * root;
    block_sz = sz / root;

    inputA.resize(sz * sz);
    inputB.resize(sz * sz);
    output = std::vector<double>(matrix_size * matrix_size, 0.0);

    std::vector<double> bufferA(matrix_size * matrix_size);
    std::vector<double> bufferB(matrix_size * matrix_size);

    std::memcpy(bufferA.data(), task_data->inputs[0], matrix_size * matrix_size * sizeof(double));
    std::memcpy(bufferB.data(), task_data->inputs[1], matrix_size * matrix_size * sizeof(double));

    double *it1 = bufferA.data();
    double *it2 = bufferB.data();
    //    double *it1 = (double *)(task_data->inputs[0]);
    //   double *it2 = (double *)(task_data->inputs[1]);
    for (int i = 0; i < matrix_size; ++i) {
      std::copy(it1 + i * matrix_size, it1 + (i + 1) * matrix_size, inputA.begin() + i * sz);
      std::copy(it2 + i * matrix_size, it2 + (i + 1) * matrix_size, inputB.begin() + i * sz);
    }
  }
  root = (int)sqrt(world.size());
  return true;
}

bool shkurinskaya_e_fox_mat_mul_mpi::FoxMatMulMPI::ValidationImpl() {
  if (world.rank() == 0) {
    root = (int)sqrt(world.size());
    return (task_data->inputs_count[0] > 0) && (task_data->inputs_count[0] == task_data->outputs_count[0]) &&
           (root * root == world.size());
  }
  return true;
}

bool shkurinskaya_e_fox_mat_mul_mpi::FoxMatMulMPI::RunImpl() {
  // share size and block size
  boost::mpi::broadcast(world, block_sz, 0);
  boost::mpi::broadcast(world, sz, 0);

  int block_matix_size = block_sz * block_sz;
  std::vector<double> local_res(block_matix_size, 0.0), left_block(block_matix_size, 0.0),
      right_block(block_matix_size, 0.0), left_to_send(block_matix_size, 0.0), right_to_send(block_matix_size, 0.0);

  // share blocks
  if (world.rank() == 0) {
    for (int i = 0; i < block_sz; ++i) {
      for (int j = 0; j < block_sz; ++j) {
        left_block[i * block_sz + j] = inputA[i * sz + j];
      }
    }

    for (int i = 0; i < block_sz; ++i) {
      for (int j = 0; j < block_sz; ++j) {
        right_block[i * block_sz + j] = inputB[i * sz + j];
      }
    }

    for (int proc = 1; proc < world.size(); ++proc) {
      int local_color = proc / root, local_key = proc % root;

      for (int i = 0; i < block_sz; i++) {
        for (int j = 0; j < block_sz; j++) {
          left_to_send[i * block_sz + j] = inputA[(local_color * block_sz + i) * sz + local_key * block_sz + j];
        }
      }

      for (int i = 0; i < block_sz; i++) {
        for (int j = 0; j < block_sz; j++) {
          right_to_send[i * block_sz + j] = inputB[(local_color * block_sz + i) * sz + local_key * block_sz + j];
        }
      }

      world.send(proc, 0, left_to_send);
      world.send(proc, 0, right_to_send);
    }
  } else {
    world.recv(0, 0, left_block);
    world.recv(0, 0, right_block);
  }

  std::vector<double> recv_block_result(block_sz * block_sz, 0.0);

  int color = world.rank() / root, key = world.rank() % root;
  // create row communicator to broadcast A~i~(i + k)
  boost::mpi::communicator row_comm = world.split(color, key);

  std::vector<double> temp = left_block, temp_output(sz * sz, 0.0);

  // Fox's algorithm (root steps)
  for (int it = 0; it < root; ++it) {
    // (color + key + it) % root  equals A~i~(i + k)
    if (((color + key + it) % root) == 0) {
      left_block = temp;
    }

    // Broadcast the local block to all ranks in the row
    boost::mpi::broadcast(row_comm, left_block.data(), left_block.size(), (root - color - it + root) % root);

    // Matrix multiplication
    if (world.rank() == 0) {
      for (int i = 0; i < block_sz; ++i) {
        for (int j = 0; j < block_sz; ++j) {
          for (int k = 0; k < block_sz; ++k) {
            temp_output[i * sz + j] += left_block[i * block_sz + k] * right_block[k * block_sz + j];
          }
        }
      }
    } else {
      for (int i = 0; i < block_sz; ++i) {
        for (int j = 0; j < block_sz; ++j) {
          for (int k = 0; k < block_sz; ++k) {
            local_res[i * block_sz + j] += left_block[i * block_sz + k] * right_block[k * block_sz + j];
          }
        }
      }
    }
    // do not need to send/recv right block
    if (it == root - 1) {
      break;
    }

    int prev = ((world.rank() / root - 1 + root) % root) * root + key,
        next = ((world.rank() / root + 1) % root) * root + key;

    if (color == 0) {
      std::vector<double> tmp(right_block.size());
      world.recv(prev, 0, tmp);
      world.send(next, 0, right_block);
      right_block = tmp;
    } else {
      world.send(next, 0, right_block);
      world.recv(prev, 0, right_block);
    }
  }

  // Gather results on rank 0
  if (world.rank() == 0) {
    for (int proc = 1; proc < world.size(); proc++) {
      world.recv(proc, 0, local_res);
      int local_color = proc / root, local_key = proc % root;
      int mergin_x = block_sz * local_color, mergin_y = block_sz * local_key;

      for (int i = 0; i < block_sz; i++) {
        for (int j = 0; j < block_sz; j++) {
          temp_output[(mergin_x + i) * sz + j + mergin_y] = local_res[i * block_sz + j];
        }
      }
    }
  } else {
    world.send(0, 0, local_res);
  }

  if (world.rank() == 0) {
    for (int i = 0; i < matrix_size; ++i) {
      for (int j = 0; j < matrix_size; ++j) {
        output[i * matrix_size + j] = temp_output[i * sz + j];
      }
    }
  }
  return true;
}

bool shkurinskaya_e_fox_mat_mul_mpi::FoxMatMulMPI::PostProcessingImpl() {
  if (world.rank() == 0) {
    std::memcpy(task_data->outputs[0], output.data(), matrix_size * matrix_size * sizeof(double));
  }
  return true;
}
