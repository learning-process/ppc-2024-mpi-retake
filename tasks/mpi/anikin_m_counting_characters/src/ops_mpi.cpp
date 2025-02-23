// Anikin Maksim 2025
#include "mpi/anikin_m_counting_characters/include/ops_mpi.hpp"

#include <boost/mpi/collectives.hpp>
#include <cmath>
#include <functional>
#include <mpi.h>
#include <random>
#include <string>
#include <vector>

void anikin_m_counting_characters_mpi::CreateDataVector(std::vector<char> *invec, const std::string& str) {
  for (auto a : str) {
    invec->push_back(a);
  }
}

void anikin_m_counting_characters_mpi::CreateRanddataVector(std::vector<char> *invec, int count) {
  for (int i = 0; i < count; i++) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis('A', 'Z');
    char random_har_ar = static_cast<char>(dis(gen));
    invec->push_back(random_har_ar);
  }
}

bool anikin_m_counting_characters_mpi::TestTaskMPI::ValidationImpl() {
  if (world_.rank() == 0) {
    return (task_data->inputs.size() == 2) && (task_data->inputs_count.size() == 2) && (task_data->outputs.size() == 1);
  }
  return true;
}
bool anikin_m_counting_characters_mpi::TestTaskMPI::PreProcessingImpl() {
  int input1_size = static_cast<int>(task_data->inputs_count[0]);
  int input2_size = static_cast<int>(task_data->inputs_count[1]);

  res_ = input1_size - input2_size;

  if (res_ <= 0) {
    auto *inlarge_ptr = reinterpret_cast<char *>(task_data->inputs[1]);
    input_1_ = std::vector<char>(inlarge_ptr, inlarge_ptr + input2_size);

    auto *insmall_ptr = reinterpret_cast<char *>(task_data->inputs[0]);
    input_2_ = std::vector<char>(insmall_ptr, insmall_ptr + input1_size);

    res_ = abs(res_);
  } else {
    auto *inlarge_ptr = reinterpret_cast<char *>(task_data->inputs[0]);
    input_1_ = std::vector<char>(inlarge_ptr, inlarge_ptr + input1_size);

    auto *insmall_ptr = reinterpret_cast<char *>(task_data->inputs[1]);
    input_2_ = std::vector<char>(insmall_ptr, insmall_ptr + input2_size);
  }
  return true;
}

bool anikin_m_counting_characters_mpi::TestTaskMPI::RunImpl() {
  std::vector<int> counts(world_.size());
  std::vector<int> displs(world_.size());
  int base = static_cast<int>(input_2_.size()) / world_.size();
  int rem = static_cast<int>(input_2_.size()) % world_.size();

  for (int i = 0; i < world_.size(); ++i) {
    counts[i] = (i < rem) ? (base + 1) : base;
    displs[i] = (i == 0) ? 0 : displs[i - 1] + counts[i - 1];
  }

  int local_size = counts[world_.rank()];
  int local_res = 0;
  std::vector<char> local_data(local_size);
  std::vector<char> cmp_local_data(local_size);

  MPI_Scatterv(input_2_.data(), counts.data(), displs.data(), MPI_CHAR, local_data.data(), local_size, MPI_CHAR, 0,
               world_);
  MPI_Scatterv(input_1_.data(), counts.data(), displs.data(), MPI_CHAR, cmp_local_data.data(), local_size, MPI_CHAR, 0,
               world_);
  auto b = local_data.begin();
  for (auto a : cmp_local_data) {
    if ((a) != (*b)) {
      local_res++;
    }
    b++;
  }
  int all_res = 0;
  boost::mpi::reduce(world_, local_res, all_res, std::plus(), 0);
  if (world_.rank() == 0) {
    res_ = res_ + all_res;
  }
  MPI_Bcast(&res_, 1, MPI_INT, 0, world_);
  return true;
}

bool anikin_m_counting_characters_mpi::TestTaskMPI::PostProcessingImpl() {
  reinterpret_cast<int *>(task_data->outputs[0])[0] = res_;
  return true;
}