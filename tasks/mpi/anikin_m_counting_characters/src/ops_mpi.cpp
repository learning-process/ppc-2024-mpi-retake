// Anikin Maksim 2025
#include "mpi/anikin_m_counting_characters/include/ops_mpi.hpp"

#include <cmath>
#include <vector>
#include <random>

void anikin_m_counting_characters_mpi::create_data_vector(std::vector<char>* invec, std::string str) {
  for (auto a : str) {
    invec->push_back(a);
  }
}

void anikin_m_counting_characters_mpi::create_randdata_vector(std::vector<char>* invec, int count) {
  for (int i = 0; i < count; i++) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis('A', 'Z');
    char randomChar = static_cast<char>(dis(gen));
    invec->push_back(randomChar);
  }
}

bool anikin_m_counting_characters_mpi::TestTaskMPI::ValidationImpl() { 
  if (world_.rank() == 0) {
      return (task_data->inputs.size() == 2) && 
             (task_data->inputs_count.size() == 2) && 
             (task_data->outputs.size() == 1);
  }
  return true;
}
bool anikin_m_counting_characters_mpi::TestTaskMPI::PreProcessingImpl() {
    int input1_size = task_data->inputs_count[0];
    int input2_size = task_data->inputs_count[1];

    res = input1_size - input2_size;

    if (res <= 0) {
      auto *inlarge_ptr = reinterpret_cast<char *>(task_data->inputs[1]);
      input_1 = std::vector<char>(inlarge_ptr, inlarge_ptr + input2_size);

      auto *insmall_ptr = reinterpret_cast<char *>(task_data->inputs[0]);
      input_2 = std::vector<char>(insmall_ptr, insmall_ptr + input1_size);

      res = abs(res);
    } else {
      auto *inlarge_ptr = reinterpret_cast<char *>(task_data->inputs[0]);
      input_1 = std::vector<char>(inlarge_ptr, inlarge_ptr + input1_size);

      auto *insmall_ptr = reinterpret_cast<char *>(task_data->inputs[1]);
      input_2 = std::vector<char>(insmall_ptr, insmall_ptr + input2_size);
    }
    return true;
}


bool anikin_m_counting_characters_mpi::TestTaskMPI::RunImpl() {
  std::vector<int> counts(world_.size());
  std::vector<int> displs(world_.size());
  int base = input_2.size() / world_.size();
  int rem = input_2.size() % world_.size();

  for (int i = 0; i < world_.size(); ++i) {
    counts[i] = (i < rem) ? (base + 1) : base;
    displs[i] = (i == 0) ? 0 : displs[i - 1] + counts[i - 1];
  }

  int local_size = counts[world_.rank()];
  int local_res = 0;
  std::vector<char> local_data(local_size);
  std::vector<char> cmp_local_data(local_size);

  MPI_Scatterv(input_2.data(), counts.data(), displs.data(),
               MPI_CHAR, local_data.data(), local_size, 
               MPI_CHAR, 0, MPI_COMM_WORLD);
  MPI_Scatterv(input_1.data(), counts.data(), displs.data(), 
               MPI_CHAR, cmp_local_data.data(), local_size,
               MPI_CHAR, 0, MPI_COMM_WORLD);
  auto b = local_data.begin();
  for (auto a : cmp_local_data) {
    if ((a) != (*b)) local_res++;
    b++;
  }
  int all_res;
  boost::mpi::reduce(world_, local_res, all_res, std::plus(), 0);
  if (world_.rank() == 0) {
    res = res + all_res;
  }
  return true;
}

bool anikin_m_counting_characters_mpi::TestTaskMPI::PostProcessingImpl() {
  if (world_.rank() == 0) {
    reinterpret_cast<int *>(task_data->outputs[0])[0] = res;
  }
  return true;
}