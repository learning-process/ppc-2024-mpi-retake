#include "mpi/deryabin_m_cannons_algorithm/include/ops_mpi.hpp"

#include <cmath>
#include <vector>

bool deryabin_m_cannons_algorithm_mpi::CannonsAlgorithmMPITaskSequential::PreProcessingImpl() {
  input_matrix_A_ = std::vector<double>(task_data->inputs_count[0]);
  input_matrix_B_ = std::vector<double>(task_data->inputs_count[1]);
  auto* tmp_ptr_a = reinterpret_cast<double*>(task_data->inputs[0]);
  auto* tmp_ptr_b = reinterpret_cast<double*>(task_data->inputs[1]);
  std::copy(tmp_ptr_a, tmp_ptr_a + task_data->inputs_count[0], input_matrix_A_.begin());
  std::copy(tmp_ptr_b, tmp_ptr_b + task_data->inputs_count[1], input_matrix_B_.begin());
  output_matrix_C_ = std::vector<double>(input_matrix_A_.size());
  return true;
}

bool deryabin_m_cannons_algorithm_mpi::CannonsAlgorithmMPITaskSequential::ValidationImpl() {
  return task_data->inputs_count[0] == task_data->inputs_count[1] &&
         task_data->inputs_count[1] == pow((unsigned short)sqrt(task_data->inputs_count[0]), 2) &&
         task_data->outputs_count[0] == 1;
}

bool deryabin_m_cannons_algorithm_mpi::CannonsAlgorithmMPITaskSequential::RunImpl() {
  unsigned short i = 0;
  unsigned short j = 0;
  unsigned short count = 0;
  auto dimension = sqrt(static_cast<unsigned short>(input_matrix_A_.size()));
  while (i != dimension) {
    j = 0;
    while (j != dimension) {
      count = 0;
      while (count != dimension) {
        output_matrix_C_[(i * dimension) + j] +=
            input_matrix_A_[(i * dimension) + count] * input_matrix_B_[(count * dimension) + j];
        count++;
      }
      j++;
    }
    i++;
  }
  return true;
}

bool deryabin_m_cannons_algorithm_mpi::CannonsAlgorithmMPITaskSequential::PostProcessingImpl() {
  reinterpret_cast<std::vector<double>*>(task_data->outputs[0])[0] = output_matrix_C_;
  return true;
}

bool deryabin_m_cannons_algorithm_mpi::CannonsAlgorithmMPITaskParallel::PreProcessingImpl() {
  if (world.rank() == 0) {
    input_matrix_A_ = std::vector<double>(task_data->inputs_count[0]);
    input_matrix_B_ = std::vector<double>(task_data->inputs_count[1]);
    auto* tmp_ptr_a = reinterpret_cast<double*>(task_data->inputs[0]);
    auto* tmp_ptr_b = reinterpret_cast<double*>(task_data->inputs[1]);
    std::copy(tmp_ptr_a, tmp_ptr_a + task_data->inputs_count[0], input_matrix_A_.begin());
    std::copy(tmp_ptr_b, tmp_ptr_b + task_data->inputs_count[1], input_matrix_B_.begin());
  }
  return true;
}

bool deryabin_m_cannons_algorithm_mpi::CannonsAlgorithmMPITaskParallel::ValidationImpl() {
  if (world.rank() == 0) {
    return task_data->inputs_count[0] == task_data->inputs_count[1] &&
           task_data->inputs_count[1] == pow((unsigned short)sqrt(task_data->inputs_count[0]), 2) &&
           task_data->outputs_count[0] == 1;
  }
  return true;
}

bool deryabin_m_cannons_algorithm_mpi::CannonsAlgorithmMPITaskParallel::RunImpl() {
  unsigned short i = 0;
  unsigned short j = 0;
  unsigned short k = 0;
  unsigned short dimension = 0;
  if (world.size() == 1 || world.size() != pow(sqrt(static_cast<unsigned short>(world.size())), 2) ||
      sqrt(static_cast<unsigned short>(input_matrix_A_.size())) % sqrt(static_cast<unsigned short>(world.size())) != 0) {
    if (world.rank() == 0) {
      dimension = sqrt(static_cast<unsigned short>(input_matrix_A_.size()));
      output_matrix_C_ = std::vector<double>(dimension * dimension);
      while (i != dimension) {
        j = 0;
        while (j != dimension) {
          k = 0;
          while (k != dimension) {
            output_matrix_C_[(i * dimension) + j] += input_matrix_A_[(i * dimension) + k] * input_matrix_B_[(k * dimension) + j];
            k++;
          }
          j++;
        }
        i++;
      }
    }
  } else {
    unsigned short block_dimension = 0;
    unsigned short block_rows_columns = 0;
    if (world.rank() == 0) {
      dimension = (unsigned short)sqrt(input_matrix_A_.size());
      block_rows_columns = (unsigned short)sqrt(world.size());
      block_dimension = dimension / block_rows_columns;
    }
    boost::mpi::broadcast(world, dimension, 0);
    output_matrix_C_ = std::vector<double>(dimension * dimension);
    boost::mpi::broadcast(world, block_dimension, 0);
    local_input_matrix_A_ = std::vector<double>(block_dimension * block_dimension);
    local_input_matrix_B_ = std::vector<double>(block_dimension * block_dimension);
    local_output_matrix_C_ = std::vector<double>(block_dimension * block_dimension);
    boost::mpi::broadcast(world, block_rows_columns, 0);
    if (world.rank() == 0) {
      k = 0;
      while (k != block_dimension) {
        std::copy(input_matrix_A_.data() + (k * dimension), input_matrix_A_.data() + (k * dimension) + block_dimension,
                  local_input_matrix_A_.begin() + (k * block_dimension));
        std::copy(input_matrix_B_.data() + (k * dimension), input_matrix_B_.data() + (k * dimension) + block_dimension,
                  local_input_matrix_B_.begin() + (k * block_dimension));
        k++;
      }
      while (i != block_rows_columns) {
        j = 0;
        while (j != block_rows_columns) {
          if (i != 0 || j != 0) {
            k = 0;
            while (k != block_dimension) {
              if (i == 0) {
                world.send((i * block_rows_columns) + j, 0,
                           input_matrix_A_.data() + ((i * block_dimension + k) * dimension) + (j * block_dimension),
                           block_dimension);
              } else {
                if ((i * block_rows_columns) + j - i < i * block_rows_columns) {
                  world.send((i * block_rows_columns) + j + block_rows_columns - i, 0,
                             input_matrix_A_.data() + ((i * block_dimension + k) * dimension) + (j * block_dimension),
                             block_dimension);
                } else {
                  world.send((i * block_rows_columns) + j - i, 0,
                             input_matrix_A_.data() + ((i * block_dimension + k) * dimension) + (j * block_dimension),
                             block_dimension);
                }
              }
              if (j == 0) {
                world.send((i * block_rows_columns) + j, 1,
                           input_matrix_B_.data() + ((i * block_dimension + k) * dimension) + (j * block_dimension),
                           block_dimension);
              } else {
                if ((i - j) * block_rows_columns + j < 0) {
                  world.send(((i + block_rows_columns - j) * block_rows_columns) + j, 1,
                             input_matrix_B_.data() + ((i * block_dimension + k) * dimension) + (j * block_dimension),
                             block_dimension);
                } else {
                  world.send(((i - j) * block_rows_columns) + j, 1,
                             input_matrix_B_.data() + (i * block_dimension + k) * dimension + (j * block_dimension),
                             block_dimension);
                }
              }
              k++;
            }
          }
          j++;
        }
        i++;
      }
    } else {
      k = 0;
      while (k != block_dimension) {
        world.recv(0, 0, local_input_matrix_A_.data() + (k * block_dimension), block_dimension);
        world.recv(0, 1, local_input_matrix_B_.data() + (k * block_dimension), block_dimension);
        k++;
      }
    }
    i = 0;
    while (i != block_dimension) {
      j = 0;
      while (j != block_dimension) {
        k = 0;
        while (k != block_dimension) {
          local_output_matrix_C_[(i * block_dimension) + j] +=
              local_input_matrix_A_[(i * block_dimension) + k] * local_input_matrix_B_[(k * block_dimension) + j];
          k++;
        }
        j++;
      }
      i++;
    }
    unsigned short p = 1;
    while (p != block_rows_columns) {
      if (block_rows_columns != 0 && world.rank() % block_rows_columns == 0) {
        world.send(world.rank() + block_rows_columns - 1, 2, local_input_matrix_A_.data(),
                   block_dimension * block_dimension);
      } else {
        world.send(world.rank() - 1, 3, local_input_matrix_A_.data(), block_dimension * block_dimension);
      }
      if (world.rank() < block_rows_columns) {
        world.send(world.rank() + block_rows_columns * (block_rows_columns - 1), 4, local_input_matrix_B_.data(),
                   block_dimension * block_dimension);
      } else {
        world.send(world.rank() - block_rows_columns, 5, local_input_matrix_B_.data(),
                   block_dimension * block_dimension);
      }
      if (block_rows_columns != 0 && (world.rank() + 1) % block_rows_columns == 0) {
        world.recv(world.rank() - block_rows_columns + 1, 2, local_input_matrix_A_.data(),
                   block_dimension * block_dimension);
      } else {
        world.recv(world.rank() + 1, 3, local_input_matrix_A_.data(), block_dimension * block_dimension);
      }
      if (world.rank() >= block_rows_columns * (block_rows_columns - 1)) {
        world.recv(world.rank() - block_rows_columns * (block_rows_columns - 1), 4, local_input_matrix_B_.data(),
                   block_dimension * block_dimension);
      } else {
        world.recv(world.rank() + block_rows_columns, 5, local_input_matrix_B_.data(),
                   block_dimension * block_dimension);
      }
      i = 0;
      while (i != block_dimension) {
        j = 0;
        while (j != block_dimension) {
          k = 0;
          while (k != block_dimension) {
            local_output_matrix_C_[(i * block_dimension) + j] +=
                local_input_matrix_A_[(i * block_dimension) + k] * local_input_matrix_B_[(k * block_dimension) + j];
            k++;
          }
          j++;
        }
        i++;
      }
      p++;
    }
    if (world.rank() != 0) {
      for (unsigned short block_row = 0; block_row < block_dimension; ++block_row) {
        world.send(0, 0, local_output_matrix_C_.data() + (block_row * block_dimension), block_dimension);
      }
    } else {
      for (unsigned short proc = 1; proc < world.size(); ++proc) {
        for (unsigned short block_row = 0; block_row < block_dimension; ++block_row) {
          std::copy(
              local_output_matrix_C_.begin() + (block_row * block_dimension),
              local_output_matrix_C_.begin() + ((block_row + 1) * block_dimension),
              output_matrix_C_.begin() + (((world.rank() / block_rows_columns) * block_dimension) + (block_row * dimension) +
                                         ((world.rank() % block_rows_columns) * block_dimension)));
          world.recv(proc, 0,
                     output_matrix_C_.data() + ((((proc / block_rows_columns) * block_dimension) + block_row) * dimension) +
                         ((proc % block_rows_columns) * block_dimension),
                     block_dimension);
        }
      }
    }
  }
  return true;
}

bool deryabin_m_cannons_algorithm_mpi::CannonsAlgorithmMPITaskParallel::PostProcessingImpl() {
  if (world.rank() == 0) {
    reinterpret_cast<std::vector<double>*>(task_data->outputs[0])[0] = output_matrix_C_;
  }
  return true;
}
