#include "mpi/karaseva_e_binaryimage/include/ops_mpi.hpp"

#include <algorithm>
#include <boost/mpi/collectives.hpp>
#include <boost/mpi/collectives/broadcast.hpp>
#include <boost/mpi/collectives/gather.hpp>
#include <boost/mpi/collectives/gatherv.hpp>
#include <boost/mpi/collectives/scatterv.hpp>
#include <cstddef>
#include <map>
#include <numeric>
#include <set>
#include <sstream>
#include <string>
#include <vector>

// Function to get the root label of a given label from the label equivalence map.
int karaseva_e_binaryimage_mpi::GetRootLabel(std::map<int, std::set<int>>& label_parent_map, int label) {
  auto search = label_parent_map.find(label);
  if (search != label_parent_map.end()) {
    return *search->second.begin();
  }

  return label;
}

// Function to propagate label equivalences, ensuring all equivalent labels are grouped together.
void karaseva_e_binaryimage_mpi::PropagateLabelEquivalences(std::map<int, std::set<int>>& label_parent_map) {
  for (auto& entry : label_parent_map) {
    for (auto value : entry.second) {
      label_parent_map[value].insert(entry.second.begin(), entry.second.end());
    }
  }
}

// Function to update labels in the labeled image with a continuous sequence of numbers.
void karaseva_e_binaryimage_mpi::UpdateLabels(std::vector<int>& labeled_image, int rows, int cols) {
  std::map<int, int> label_mapping;
  int next_label = 2;
  for (int x = 0; x < rows; x++) {
    for (int y = 0; y < cols; y++) {
      int idx = (x * cols) + y;
      if (labeled_image[idx] > 1) {
        int final_label = 0;
        auto label_search = label_mapping.find(labeled_image[idx]);
        if (label_search == label_mapping.end()) {
          final_label = next_label;
          label_mapping[labeled_image[idx]] = next_label++;
        } else {
          final_label = label_search->second;
        }

        labeled_image[idx] = final_label;
      }
    }
  }
}

// Function to merge two labels into the same equivalence group.
void karaseva_e_binaryimage_mpi::UnionLabels(std::map<int, std::set<int>>& label_parent_map, int new_label,
                                             int neighbour_label) {
  if (new_label == neighbour_label) {
    return;
  }

  auto search1 = label_parent_map.find(new_label);
  auto search2 = label_parent_map.find(neighbour_label);

  if (search1 == label_parent_map.end() && search2 == label_parent_map.end()) {
    label_parent_map[new_label].insert(neighbour_label);
    label_parent_map[new_label].insert(new_label);
    label_parent_map[neighbour_label].insert(new_label);
    label_parent_map[neighbour_label].insert(neighbour_label);
  } else if (search1 != label_parent_map.end() && search2 == label_parent_map.end()) {
    label_parent_map[new_label].insert(neighbour_label);
    label_parent_map[neighbour_label] = label_parent_map[new_label];
  } else if (search1 == label_parent_map.end() && search2 != label_parent_map.end()) {
    label_parent_map[neighbour_label].insert(new_label);
    label_parent_map[new_label] = label_parent_map[neighbour_label];
  } else {
    std::set<int> temp_set = label_parent_map[new_label];
    label_parent_map[new_label].insert(label_parent_map[neighbour_label].begin(),
                                       label_parent_map[neighbour_label].end());
    label_parent_map[neighbour_label].insert(label_parent_map[new_label].begin(), label_parent_map[new_label].end());
  }
}
// NOLINTBEGIN
// Function to perform connected-component labeling using a sequential scan approach.
void karaseva_e_binaryimage_mpi::Labeling(std::vector<int>& input_image, std::vector<int>& labeled_image, int rows,
                                          int cols, int min_label, std::map<int, std::set<int>>& label_parent_map) {
  int current_label = min_label;
  int dx[] = {-1, 0, -1};
  int dy[] = {0, -1, 1};

  for (int x = 0; x < rows; x++) {
    for (int y = 0; y < cols; y++) {
      int position = (x * cols) + y;
      if (input_image[position] == 0 || labeled_image[position] > 1) {
        std::vector<int> neighbors;

        for (int i = 0; i < 3; i++) {
          int nx = x + dx[i];
          int ny = y + dy[i];
          int tmp_pos = (nx * cols) + ny;
          if (nx >= 0 && nx < rows && ny >= 0 && ny < cols && (labeled_image[tmp_pos] > 1)) {
            neighbors.push_back(labeled_image[tmp_pos]);
          }
        }

        if (neighbors.empty() && labeled_image[position] != 0) {
          labeled_image[position] = current_label;
          current_label++;
        } else {
          int min_neighbor_label = *std::ranges::min_element(neighbors.begin(), neighbors.end());
          labeled_image[position] = min_neighbor_label;

          for (int label : neighbors) {
            UnionLabels(label_parent_map, label, min_neighbor_label);
          }
        }
      }
    }
  }

  PropagateLabelEquivalences(label_parent_map);

  // Update labels in the image to reflect the root labels from the equivalence map
  for (int x = 0; x < rows; x++) {
    for (int y = 0; y < cols; y++) {
      int position = (x * cols) + y;
      if (labeled_image[position] > 1) {
        int root_label = GetRootLabel(label_parent_map, labeled_image[position]);
        labeled_image[position] = root_label;
      }
    }
  }
}
// NOLINTEND
bool karaseva_e_binaryimage_mpi::TestMPITaskSequential::PreProcessingImpl() {
  rows_ = static_cast<int>(task_data->inputs_count[0]);
  columns_ = static_cast<int>(task_data->inputs_count[1]);
  int pixel_count = rows_ * columns_;
  image_ = std::vector<int>(pixel_count);
  auto* input_ptr = reinterpret_cast<int*>(task_data->inputs[0]);
  std::copy(input_ptr, input_ptr + pixel_count, image_.begin());

  labeled_image_ = std::vector<int>(rows_ * columns_, 1);
  return true;
}

bool karaseva_e_binaryimage_mpi::TestMPITaskSequential::ValidationImpl() {
  int tmp_rows = static_cast<int>(task_data->inputs_count[0]);
  int tmp_columns = static_cast<int>(task_data->inputs_count[1]);
  auto* input_ptr = reinterpret_cast<int*>(task_data->inputs[0]);

  for (int x = 0; x < tmp_rows; x++) {
    for (int y = 0; y < tmp_columns; y++) {
      int pixel = input_ptr[(x * tmp_columns) + y];
      if (pixel < 0 || pixel > 1) {
        return false;
      }
    }
  }
  return tmp_rows > 0 && tmp_columns > 0 && static_cast<int>(task_data->outputs_count[0]) == tmp_rows &&
         static_cast<int>(task_data->outputs_count[1]) == tmp_columns;
}

bool karaseva_e_binaryimage_mpi::TestMPITaskSequential::RunImpl() {
  std::map<int, std::set<int>> dummy_map;
  Labeling(image_, labeled_image_, rows_, columns_, 2, dummy_map);
  UpdateLabels(labeled_image_, rows_, columns_);

  return true;
}

bool karaseva_e_binaryimage_mpi::TestMPITaskSequential::PostProcessingImpl() {
  auto* output_ptr = reinterpret_cast<int*>(task_data->outputs[0]);
  std::ranges::copy(labeled_image_.begin(), labeled_image_.end(), output_ptr);
  return true;
}

bool karaseva_e_binaryimage_mpi::TestMPITaskParallel::PreProcessingImpl() {
  if (world_.rank() == 0) {
    rows_ = static_cast<int>(task_data->inputs_count[0]);
    columns_ = static_cast<int>(task_data->inputs_count[1]);
    int pixel_count = rows_ * columns_;
    image_ = std::vector<int>(pixel_count);
    auto* input_ptr = reinterpret_cast<int*>(task_data->inputs[0]);
    std::ranges::copy(input_ptr, input_ptr + pixel_count, image_.begin());

    labeled_image_ = std::vector<int>(rows_ * columns_, 1);
  }

  return true;
}

bool karaseva_e_binaryimage_mpi::TestMPITaskParallel::ValidationImpl() {
  if (world_.rank() == 0) {
    int tmp_rows = static_cast<int>(task_data->inputs_count[0]);
    int tmp_columns = static_cast<int>(task_data->inputs_count[1]);
    auto* input_ptr = reinterpret_cast<int*>(task_data->inputs[0]);

    for (int x = 0; x < tmp_rows; x++) {
      for (int y = 0; y < tmp_columns; y++) {
        int pixel = input_ptr[(x * tmp_columns) + y];
        if (pixel < 0 || pixel > 1) {
          return false;
        }
      }
    }
    return tmp_rows > 0 && tmp_columns > 0 && static_cast<int>(task_data->outputs_count[0]) == tmp_rows &&
           static_cast<int>(task_data->outputs_count[1]) == tmp_columns;
  }
  return true;
}

// Store the size of the set first, followed by its elements
void karaseva_e_binaryimage_mpi::SaveLabelSetToStream(std::ostringstream& oss, const std::set<int>& label_set) {
  oss << label_set.size() << " ";
  for (const auto& item : label_set) {
    oss << item << " ";
  }
}

// Read the size of the set and populate it with values from the stream
void karaseva_e_binaryimage_mpi::LoadLabelSetFromStream(std::istringstream& iss, std::set<int>& label_set) {
  size_t size = 0;
  iss >> size;
  label_set.clear();
  for (size_t i = 0; i < size; ++i) {
    int item = 0;
    iss >> item;
    label_set.insert(item);
  }
}

// Store the size of the map first, followed by each key and its corresponding set
void karaseva_e_binaryimage_mpi::SaveLabelMapToStream(std::ostringstream& oss,
                                                      const std::map<int, std::set<int>>& label_map) {
  oss << label_map.size() << " ";
  for (const auto& entry : label_map) {
    oss << entry.first << " ";
    SaveLabelSetToStream(oss, entry.second);
  }
}

// Read the size of the map and reconstruct each key-value pair
void karaseva_e_binaryimage_mpi::LoadLabelMapFromStream(std::istringstream& iss,
                                                        std::map<int, std::set<int>>& label_map) {
  size_t size = 0;
  iss >> size;
  label_map.clear();
  for (size_t i = 0; i < size; ++i) {
    int key = 0;
    iss >> key;
    std::set<int> value;
    LoadLabelSetFromStream(iss, value);
    label_map[key] = value;
  }
}

bool karaseva_e_binaryimage_mpi::TestMPITaskParallel::RunImpl() {
  boost::mpi::broadcast(world_, rows_, 0);
  boost::mpi::broadcast(world_, columns_, 0);

  std::vector<int> partition_sizes(world_.size(), rows_ / world_.size() * columns_);
  for (int i = 0; i < rows_ % world_.size(); i++) {
    partition_sizes[i] += columns_;
  }

  local_image_ = std::vector<int>(partition_sizes[world_.rank()]);
  boost::mpi::scatterv(world_, image_, partition_sizes, local_image_.data(), 0);

  std::vector<int> local_labeled_image(partition_sizes[world_.rank()], 1);
  int min_label = (100000 * world_.rank()) + 2;
  std::map<int, std::set<int>> local_parent_map;
  Labeling(local_image_, local_labeled_image, partition_sizes[world_.rank()] / columns_, columns_, min_label,
           local_parent_map);

  boost::mpi::gatherv(world_, local_labeled_image, labeled_image_.data(), partition_sizes, 0);

  std::ostringstream oss;
  SaveLabelMapToStream(oss, local_parent_map);
  std::string serialized_data = oss.str();

  std::vector<int> data_sizes(world_.size());
  int data_size = static_cast<int>(serialized_data.size());
  boost::mpi::gather(world_, data_size, data_sizes, 0);

  int buffer_size = 0;
  std::vector<char> buffer;

  if (world_.rank() == 0) {
    buffer_size = std::accumulate(data_sizes.begin(), data_sizes.end(), 0);
    buffer = std::vector<char>(buffer_size);
  }
  std::vector<char> send_data(serialized_data.begin(), serialized_data.end());
  boost::mpi::gatherv(world_, send_data, buffer.data(), data_sizes, 0);

  if (world_.rank() == 0) {
    std::map<int, std::set<int>> global_map;
    int displacement = 0;
    for (int i = 0; i < world_.size(); i++) {
      std::string map_data = std::string(buffer.begin() + displacement, buffer.begin() + displacement + data_sizes[i]);
      std::istringstream input_stream(map_data);
      std::map<int, std::set<int>> received_map;
      LoadLabelMapFromStream(input_stream, received_map);
      displacement += data_sizes[i];
      global_map.insert(received_map.begin(), received_map.end());
    }

    Labeling(image_, labeled_image_, rows_, columns_, 2, global_map);
    UpdateLabels(labeled_image_, rows_, columns_);
  }

  return true;
}

bool karaseva_e_binaryimage_mpi::TestMPITaskParallel::PostProcessingImpl() {
  if (world_.rank() == 0) {
    auto* output_ptr = reinterpret_cast<int*>(task_data->outputs[0]);
    std::ranges::copy(labeled_image_.begin(), labeled_image_.end(), output_ptr);
  }
  return true;
}