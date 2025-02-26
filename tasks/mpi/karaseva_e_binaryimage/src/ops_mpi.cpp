#include "mpi/karaseva_e_binaryimage/include/ops_mpi.hpp"

#include <iostream>
#include <map>
#include <set>
#include <sstream>
#include <vector>
// NOLINTBEGIN
// Finding the root of the label (smallest label in the object)
int karaseva_e_binaryimage_mpi::GetRootLabel(std::map<int, std::set<int>>& label_parent_map, int label) {
  auto search = label_parent_map.find(label);
  if (search != label_parent_map.end()) {
    return *search->second.begin();
  }

  return label;
}

// Correcting all the connections in the table after the first pass
void karaseva_e_binaryimage_mpi::PropagateLabelEquivalences(std::map<int, std::set<int>>& label_parent_map) {
  for (auto& entry : label_parent_map) {
    for (auto value : entry.second) {
      label_parent_map[value].insert(entry.second.begin(), entry.second.end());
    }
  }
}

// Correcting all labels in the labeled image to proper values
void karaseva_e_binaryimage_mpi::UpdateLabels(std::vector<int>& labeled_image, int rows, int cols) {
  std::map<int, int> labelMapping;
  int nextLabel = 2;
  for (int x = 0; x < rows; x++) {
    for (int y = 0; y < cols; y++) {
      int idx = x * cols + y;
      if (labeled_image[idx] > 1) {
        int finalLabel;
        auto labelSearch = labelMapping.find(labeled_image[idx]);
        if (labelSearch == labelMapping.end()) {
          finalLabel = nextLabel;
          labelMapping[labeled_image[idx]] = nextLabel++;
        } else {
          finalLabel = labelSearch->second;
        }

        labeled_image[idx] = finalLabel;
      }
    }
  }
}

// Merging two related labels
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
    std::set<int> tempSet = label_parent_map[new_label];
    label_parent_map[new_label].insert(label_parent_map[neighbour_label].begin(),
                                       label_parent_map[neighbour_label].end());
    label_parent_map[neighbour_label].insert(label_parent_map[new_label].begin(), label_parent_map[new_label].end());
  }
}

// Sequential labeling algorithm (used in each MPI process)
void karaseva_e_binaryimage_mpi::Labeling(std::vector<int>& input_image, std::vector<int>& labeled_image, int rows,
                                          int cols, int min_label, std::map<int, std::set<int>>& label_parent_map) {
  int currentLabel = min_label;
  // Directions for neighboring pixels
  int dx[] = {-1, 0, -1};
  int dy[] = {0, -1, 1};

  for (int x = 0; x < rows; x++) {
    for (int y = 0; y < cols; y++) {
      int position = x * cols + y;
      if (input_image[position] == 0 || labeled_image[position] > 1) {
        std::vector<int> neighbors;

        for (int i = 0; i < 3; i++) {
          int nx = x + dx[i];
          int ny = y + dy[i];
          int tmpPos = nx * cols + ny;
          if (nx >= 0 && nx < rows && ny >= 0 && ny < cols && (labeled_image[tmpPos] > 1)) {
            neighbors.push_back(labeled_image[tmpPos]);
          }
        }

        if (neighbors.empty() && labeled_image[position] != 0) {
          labeled_image[position] = currentLabel;
          currentLabel++;
        } else {
          int minNeighborLabel = *std::min_element(neighbors.begin(), neighbors.end());
          labeled_image[position] = minNeighborLabel;

          for (int label : neighbors) {
            UnionLabels(label_parent_map, minNeighborLabel, label);
          }
        }
      }
    }
  }

  PropagateLabelEquivalences(label_parent_map);

  for (int x = 0; x < rows; x++) {
    for (int y = 0; y < cols; y++) {
      int position = x * cols + y;
      if (labeled_image[position] > 1) {
        int rootLabel = GetRootLabel(label_parent_map, labeled_image[position]);
        labeled_image[position] = rootLabel;
      }
    }
  }
}

bool karaseva_e_binaryimage_mpi::TestMPITaskSequential::PreProcessingImpl() {
  rows = task_data->inputs_count[0];
  columns = task_data->inputs_count[1];
  int pixelCount = rows * columns;
  image_ = std::vector<int>(pixelCount);
  auto* inputPtr = reinterpret_cast<int*>(task_data->inputs[0]);
  std::copy(inputPtr, inputPtr + pixelCount, image_.begin());

  labeled_image = std::vector<int>(rows * columns, 1);
  return true;
}

bool karaseva_e_binaryimage_mpi::TestMPITaskSequential::ValidationImpl() {
  int tmpRows = task_data->inputs_count[0];
  int tmpColumns = task_data->inputs_count[1];
  auto* inputPtr = reinterpret_cast<int*>(task_data->inputs[0]);

  for (int x = 0; x < tmpRows; x++) {
    for (int y = 0; y < tmpColumns; y++) {
      int pixel = inputPtr[x * tmpColumns + y];
      if (pixel < 0 || pixel > 1) {
        return false;
      }
    }
  }
  return tmpRows > 0 && tmpColumns > 0 && static_cast<int>(task_data->outputs_count[0]) == tmpRows &&
         static_cast<int>(task_data->outputs_count[1]) == tmpColumns;
}

bool karaseva_e_binaryimage_mpi::TestMPITaskSequential::RunImpl() {
  std::map<int, std::set<int>> dummyMap;
  Labeling(image_, labeled_image, rows, columns, 2, dummyMap);
  UpdateLabels(labeled_image, rows, columns);

  return true;
}

bool karaseva_e_binaryimage_mpi::TestMPITaskSequential::PostProcessingImpl() {
  auto* outputPtr = reinterpret_cast<int*>(task_data->outputs[0]);
  std::copy(labeled_image.begin(), labeled_image.end(), outputPtr);
  return true;
}

bool karaseva_e_binaryimage_mpi::TestMPITaskParallel::PreProcessingImpl() {
  if (world.rank() == 0) {
    rows = task_data->inputs_count[0];
    columns = task_data->inputs_count[1];
    int pixelCount = rows * columns;
    image_ = std::vector<int>(pixelCount);
    auto* inputPtr = reinterpret_cast<int*>(task_data->inputs[0]);
    std::copy(inputPtr, inputPtr + pixelCount, image_.begin());

    labeled_image = std::vector<int>(rows * columns, 1);
  }

  std::cout << "Rank " << world.rank() << " - PreProcessingImpl completed" << std::endl;

  return true;
}

bool karaseva_e_binaryimage_mpi::TestMPITaskParallel::ValidationImpl() {
  if (world.rank() == 0) {
    int tmpRows = task_data->inputs_count[0];
    int tmpColumns = task_data->inputs_count[1];
    auto* inputPtr = reinterpret_cast<int*>(task_data->inputs[0]);

    for (int x = 0; x < tmpRows; x++) {
      for (int y = 0; y < tmpColumns; y++) {
        int pixel = inputPtr[x * tmpColumns + y];
        if (pixel < 0 || pixel > 1) {
          return false;
        }
      }
    }
    return tmpRows > 0 && tmpColumns > 0 && static_cast<int>(task_data->outputs_count[0]) == tmpRows &&
           static_cast<int>(task_data->outputs_count[1]) == tmpColumns;
  }
  return true;
}

void SaveLabelSetToStream(std::ostringstream& oss, const std::set<int>& labelSet) {
  oss << labelSet.size() << " ";  // Write the size of the set
  for (const auto& item : labelSet) {
    oss << item << " ";  // Write each item
  }
}

void LoadLabelSetFromStream(std::istringstream& iss, std::set<int>& labelSet) {
  size_t size;
  iss >> size;  // Read the size of the set
  labelSet.clear();
  for (size_t i = 0; i < size; ++i) {
    int item;
    iss >> item;  // Read each item
    labelSet.insert(item);
  }
}

// Custom serialization for std::map
void karaseva_e_binaryimage_mpi::SaveLabelMapToStream(std::ostringstream& oss,
                                                      const std::map<int, std::set<int>>& label_map) {
  oss << label_map.size() << " ";
  for (const auto& entry : label_map) {
    oss << entry.first << " ";
    SaveLabelSetToStream(oss, entry.second);
  }
}

// Custom deserialization for std::map
void karaseva_e_binaryimage_mpi::LoadLabelMapFromStream(std::istringstream& iss,
                                                        std::map<int, std::set<int>>& label_map) {
  size_t size;
  iss >> size;  // Read the size of the map
  label_map.clear();
  for (size_t i = 0; i < size; ++i) {
    int key;
    iss >> key;  // Read the key
    std::set<int> value;
    LoadLabelSetFromStream(iss, value);  // Deserialize the set
    label_map[key] = value;
  }
}

bool karaseva_e_binaryimage_mpi::TestMPITaskParallel::RunImpl() {
  std::cout << "Rank " << world.rank() << " - RunImpl started" << std::endl;

  boost::mpi::broadcast(world, rows, 0);
  boost::mpi::broadcast(world, columns, 0);

  std::vector<int> partitionSizes(world.size(), rows / world.size() * columns);
  for (int i = 0; i < rows % world.size(); i++) {
    partitionSizes[i] += columns;
  }

  local_image_ = std::vector<int>(partitionSizes[world.rank()]);
  boost::mpi::scatterv(world, image_, partitionSizes, local_image_.data(), 0);

  std::cout << "Rank " << world.rank() << " - Image data scattered" << std::endl;

  // Local labeling
  std::vector<int> localLabeledImage(partitionSizes[world.rank()], 1);
  int min_label = 100000 * world.rank() + 2;
  std::map<int, std::set<int>> localParentMap;
  Labeling(local_image_, localLabeledImage, partitionSizes[world.rank()] / columns, columns, min_label, localParentMap);

  boost::mpi::gatherv(world, localLabeledImage, labeled_image.data(), partitionSizes, 0);

  // Prepare and serialize label map data
  std::ostringstream oss;
  SaveLabelMapToStream(oss, localParentMap);
  std::string serializedData = oss.str();

  std::vector<int> dataSizes(world.size());
  int dataSize = static_cast<int>(serializedData.size());
  boost::mpi::gather(world, dataSize, dataSizes, 0);

  int bufferSize;
  std::vector<char> buffer;

  if (world.rank() == 0) {
    bufferSize = std::accumulate(dataSizes.begin(), dataSizes.end(), 0);
    buffer = std::vector<char>(bufferSize);
  }
  std::vector<char> sendData(serializedData.begin(), serializedData.end());
  boost::mpi::gatherv(world, sendData, buffer.data(), dataSizes, 0);

  std::cout << "Rank " << world.rank() << " - Data gathered" << std::endl;

  if (world.rank() == 0) {
    std::map<int, std::set<int>> globalMap;
    int displacement = 0;
    for (int i = 0; i < world.size(); i++) {
      std::string mapData = std::string(buffer.begin() + displacement, buffer.begin() + displacement + dataSizes[i]);
      std::istringstream inputStream(mapData);
      std::map<int, std::set<int>> receivedMap;
      LoadLabelMapFromStream(inputStream, receivedMap);
      displacement += dataSizes[i];
      globalMap.insert(receivedMap.begin(), receivedMap.end());
    }

    Labeling(image_, labeled_image, rows, columns, 2, globalMap);
    UpdateLabels(labeled_image, rows, columns);
  }

  std::cout << "Rank " << world.rank() << " - RunImpl completed" << std::endl;

  return true;
}

bool karaseva_e_binaryimage_mpi::TestMPITaskParallel::PostProcessingImpl() {
  if (world.rank() == 0) {
    auto* outputPtr = reinterpret_cast<int*>(task_data->outputs[0]);
    std::copy(labeled_image.begin(), labeled_image.end(), outputPtr);
  }
  return true;
}
// NOLINTEND