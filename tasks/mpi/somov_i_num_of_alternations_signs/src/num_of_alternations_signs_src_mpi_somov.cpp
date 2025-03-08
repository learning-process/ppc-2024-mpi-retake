#include <algorithm>
#include <boost/mpi/collectives/reduce.hpp>
#include <boost/mpi/collectives/scatter.hpp>
#include <boost/mpi/collectives/scatterv.hpp>
#include <cstring>
#include <functional>
#include <vector>

#include "mpi/somov_i_num_of_alternations_signs/include/num_of_alternations_signs_header_mpi_somov.hpp"
namespace somov_i_num_of_alternations_signs_mpi {
void CheckForAlternationSigns(const std::vector<int>& vec, int& out) {
  out = 0;
  for (int i = 0; i < static_cast<int>(vec.size()) - 1; ++i) {
    if (vec[i] * vec[i + 1] < 0) {
      ++out;
    }
  }
}
bool NumOfAlternationsSigns::PreProcessingImpl() {
  // Init vectors
  if (world_.rank() == 0) {
    sz_ = static_cast<int>(task_data->inputs_count[0]);
    input_ = std::vector<int>(sz_);
    std::ranges::copy(reinterpret_cast<int*>(task_data->inputs[0]), reinterpret_cast<int*>(task_data->inputs[0]) + sz_,
                      input_.begin());
  }
  return true;
}

bool NumOfAlternationsSigns::ValidationImpl() {
  if (world_.rank() == 0) {
    return (task_data->outputs_count[0] > 0 && task_data->inputs_count[0] > 0);
  }
  return true;
}

bool NumOfAlternationsSigns::RunImpl() {
  int id = world_.rank();
  int size = world_.size();
  output_ = 0;
  if (size == 1) {
    for (int i = 0; i < static_cast<int>(input_.size()) - 1; ++i) {
      if (input_[i] * input_[i + 1] < 0) {
        ++output_;
      }
    }
    return true;
  }
  std::vector<int> local_data;
  std::vector<int> send_counts;
  std::vector<int> displs;

  if (id == 0) {
    send_counts.resize(size);
    displs.resize(size);
    int base_size = sz_ / size;
    int remainder = sz_ % size;

    for (int i = 0; i < size; ++i) {
      send_counts[i] = base_size + (i < remainder ? 1 : 0);
    }
    displs[0] = 0;
    for (int i = 1; i < size; ++i) {
      displs[i] = displs[i - 1] + send_counts[i - 1];
    }
    for (int i = 0; i < size - 1; ++i) {
      if (displs[i] + send_counts[i] < sz_) {
        send_counts[i]++;
      }
    }
  }
  int loc_vec_sz = 0;
  scatter(world_, send_counts, loc_vec_sz, 0);
  local_data.resize(loc_vec_sz);
  scatterv(world_, input_.data(), send_counts, displs, local_data.data(), loc_vec_sz, 0);
  for (int i = 0; i < static_cast<int>(local_data.size()) - 1; ++i) {
    if (local_data[i] * local_data[i + 1] < 0) {
      ++output_;
    }
  }

  int global_output = 0;
  reduce(world_, output_, global_output, std::plus<>(), 0);

  if (id == 0) {
    output_ = global_output;
  }
  return true;
}

bool NumOfAlternationsSigns::PostProcessingImpl() {
  if (world_.rank() == 0) {
    reinterpret_cast<int*>(task_data->outputs[0])[0] = output_;
  }
  return true;
}
}  // namespace somov_i_num_of_alternations_signs_mpi
