#include "mpi/leontev_n_binary/include/ops_mpi.hpp"

#include <algorithm>
#include <boost/mpi.hpp>
#include <boost/mpi/collectives.hpp>
#include <boost/mpi/communicator.hpp>
#include <boost/serialization/vector.hpp>
#include <cassert>
#include <cstdint>
#include <iostream>
#include <queue>
#include <unordered_map>
#include <utility>
#include <vector>

namespace leontev_n_binary_mpi {

bool CompNotZero(uint32_t a, uint32_t b) {
  if (a == 0) {
    return false;
  }
  if (b == 0) {
    return true;
  }
  return a < b;
}

size_t BinarySegmentsMPI::GetIndex(size_t i, size_t j) { return i * cols_ + j; }

bool BinarySegmentsMPI::ValidationImpl() {
  if (world_.rank() == 0) {
    return !task_data->inputs.empty() && !task_data->outputs.empty() && task_data->inputs_count.size() == 2 &&
           task_data->outputs_count.size() == 2 && task_data->inputs_count[0] == task_data->outputs_count[0] &&
           task_data->inputs_count[1] == task_data->outputs_count[1];
  }
  return true;
}

bool BinarySegmentsMPI::PreProcessingImpl() {
  if (world_.rank() == 0) {
    rows_ = task_data->inputs_count[0];
    cols_ = task_data->inputs_count[1];
    input_image_.resize(rows_ * cols_);
    std::copy_n(reinterpret_cast<uint8_t*>(task_data->inputs[0]), rows_ * cols_, input_image_.begin());
  }
  return true;
}

bool BinarySegmentsMPI::RunImpl() {
  boost::mpi::broadcast(world_, rows_, 0);
  boost::mpi::broadcast(world_, cols_, 0);
  std::vector<int> send_counts(world_.size(), 0);
  std::vector<int> offsets(world_.size(), 0);
  int rows_for_proc = rows_ / world_.size();
  for (int i = 0; i < world_.size(); ++i) {
    if (i == 0) {
      send_counts[i] = (rows_for_proc + (rows_ % world_.size())) * cols_;
    } else {
      send_counts[i] = rows_for_proc * cols_;
      offsets[i] = offsets[i - 1] + send_counts[i - 1];
    }
  }
  size_t local_size = (world_.rank() == 0) ? (rows_for_proc + (rows_ % world_.size())) : rows_for_proc;
  local_image_.resize(local_size * cols_);
  boost::mpi::scatterv(world_, input_image_.data(), send_counts, offsets, local_image_.data(), local_size * cols_, 0);
  uint32_t next_label = 1 + offsets[world_.rank()];
  std::vector<uint32_t> local_labels_(local_size * cols_);
  std::unordered_map<uint32_t, uint32_t> local_label_equivalences;
  for (size_t row = 0; row < local_size; ++row) {
    for (size_t col = 0; col < cols_; ++col) {
      size_t cur_ind = GetIndex(row, col);
      if (local_image_[cur_ind] == 0) {
        continue;
      }
      uint32_t label_B = (col > 0) ? local_labels_[cur_ind - 1] : 0;
      uint32_t label_C = (row > 0) ? local_labels_[cur_ind - cols_] : 0;
      uint32_t label_D = (row > 0 && col > 0) ? local_labels_[cur_ind - cols_ - 1] : 0;
      if (label_B == 0 && label_C == 0 && label_D == 0) {
        local_labels_[cur_ind] = next_label++;
      } else {
        uint32_t min_label = std::min({label_B, label_C, label_D}, CompNotZero);
        local_labels_[cur_ind] = min_label;
        for (uint32_t label : {label_B, label_C, label_D}) {
          if (label != 0 && label != min_label && label > min_label) {
            local_label_equivalences[label] = min_label;
          }
          if (label != 0 && label != min_label && label < min_label) {
            local_label_equivalences[min_label] = label;
          }
        }
      }
    }
  }

  for (auto& label : local_labels_) {
    while (local_label_equivalences.contains(label)) {
      label = local_label_equivalences[label];
    }
  }
  if (world_.rank() == 0) labels_.resize(rows_ * cols_);
  boost::mpi::gatherv(world_, local_labels_, labels_.data(), send_counts, offsets, 0);
  if (world_.rank() == 0) {
    std::unordered_map<uint32_t, uint32_t> label_equivalences;
    for (int section = 1; section < world_.size(); ++section) {
      int border = offsets[section];
      if (border >= rows_ * cols_) {
        break;
      }
      for (size_t col = 0; col < cols_; ++col) {
        size_t cur_ind = border + col;
        if (labels_[cur_ind] == 0) {
          continue;
        }
        uint32_t label_B = (col > 0) ? labels_[cur_ind - 1] : 0;
        uint32_t label_C = labels_[cur_ind - cols_];
        uint32_t label_D = (col > 0) ? labels_[cur_ind - cols_ - 1] : 0;
        if (label_B != 0 || label_C != 0 || label_D != 0) {
          uint32_t min_label = std::min({label_B, label_C, label_D}, CompNotZero);
          labels_[cur_ind] = min_label;
          for (uint32_t label2 : {label_B, label_C, label_D}) {
            if (label2 != 0 && label2 != min_label && label2 > min_label) {
              label_equivalences[label2] = min_label;
            } else if (label2 != 0 && label2 != min_label && label2 < min_label) {
              label_equivalences[min_label] = label2;
            }
          }
        }
      }
    }
    for (auto const& i : label_equivalences) {
      std::cout << i.first << " " << i.second << std::endl;
    }
    for (auto& label : labels_) {
      while (label_equivalences.contains(label)) {
        label = label_equivalences[label];
      }
    }
    std::vector<size_t> arrived(rows_ * cols_ + 1, 0);
    size_t cur_mark = 1;
    for (size_t i = 0; i < rows_ * cols_; i++) {
      if (labels_[i] != 0) {
        if (arrived[labels_[i]] != 0) {
          labels_[i] = arrived[labels_[i]];
        } else {
          labels_[i] = arrived[labels_[i]] = cur_mark++;
        }
      }
    }
  }
  return true;
}

bool BinarySegmentsMPI::PostProcessingImpl() {
  if (world_.rank() == 0) {
    std::copy_n(labels_.data(), rows_ * cols_, reinterpret_cast<uint32_t*>(task_data->outputs[0]));
  }
  return true;
}

}  // namespace leontev_n_binary_mpi
