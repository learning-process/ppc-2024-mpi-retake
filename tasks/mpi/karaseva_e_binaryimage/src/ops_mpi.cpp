#include "mpi/karaseva_e_binaryimage/include/ops_mpi.hpp"

#include <iostream>
#include <map>
#include <set>
#include <sstream>
#include <vector>

// Finds the root label of a given label in the label connection map.
int karaseva_e_binaryimage_mpi::FindRootLabel(std::map<int, std::set<int>>& label_connection_map, int label) {
  auto it = label_connection_map.find(label);
  if (it != label_connection_map.end()) {
    return *it->second.begin();
  }
  return label;
}

// Combines two connected labels in the label connection map.
void karaseva_e_binaryimage_mpi::CombineLabels(std::map<int, std::set<int>>& label_connection_map, int label1,
                                               int label2) {
  if (label1 == label2) return;

  auto it1 = label_connection_map.find(label1);
  auto it2 = label_connection_map.find(label2);

  if (it1 == label_connection_map.end() && it2 == label_connection_map.end()) {
    Createnew_labelConnection(label_connection_map, label1, label2);
  } else if (it1 != label_connection_map.end() && it2 == label_connection_map.end()) {
    ConnectWithexisting_label(label_connection_map, label1, label2);
  } else if (it1 == label_connection_map.end() && it2 != label_connection_map.end()) {
    ConnectWithexisting_label(label_connection_map, label2, label1);
  } else {
    MergeLabelConnections(label_connection_map, label1, label2);
  }
}

// Connects a new label with an existing label in the map.
void karaseva_e_binaryimage_mpi::ConnectWithexisting_label(std::map<int, std::set<int>>& label_connection_map,
                                                           int existing_label, int newLabel) {
  label_connection_map[existing_label].insert(newLabel);
  label_connection_map[newLabel] = label_connection_map[existing_label];
}

// Creates a new connection between two labels in the label connection map.
void karaseva_e_binaryimage_mpi::Createnew_labelConnection(std::map<int, std::set<int>>& label_connection_map,
                                                           int label1, int label2) {
  label_connection_map[label1].insert(label2);
  label_connection_map[label1].insert(label1);
  label_connection_map[label2].insert(label1);
  label_connection_map[label2].insert(label2);
}

// Merges two label connections into a single set of connections.
void karaseva_e_binaryimage_mpi::MergeLabelConnections(std::map<int, std::set<int>>& label_connection_map, int label1,
                                                       int label2) {
  std::set<int> mergedLabels = label_connection_map[label1];
  label_connection_map[label1].insert(label_connection_map[label2].begin(), label_connection_map[label2].end());
  label_connection_map[label2].insert(mergedLabels.begin(), mergedLabels.end());
}

// Fixes label connections after the first pass by ensuring all connected labels point to the same root.
void karaseva_e_binaryimage_mpi::FixLabelConnections(std::map<int, std::set<int>>& label_connection_map) {
  for (auto& entry : label_connection_map) {
    for (auto& label : entry.second) {
      label_connection_map[label].insert(entry.second.begin(), entry.second.end());
    }
  }
}

// Corrects labels in the final labeled image by assigning final labels to each pixel.
void karaseva_e_binaryimage_mpi::CorrectLabels(std::vector<int>& labeledImage, int rows, int cols) {
  std::map<int, int> labelReassignment;
  int nextAvailableLabel = 2;

  for (int x = 0; x < rows; ++x) {
    for (int y = 0; y < cols; ++y) {
      int index = x * cols + y;
      if (labeledImage[index] > 1) {
        int finalLabel = AssignLabel(labeledImage[index], labelReassignment, nextAvailableLabel);
        labeledImage[index] = finalLabel;
      }
    }
  }
}

// Assigns a new label or finds the existing reassigned label for a given label.
int karaseva_e_binaryimage_mpi::AssignLabel(int currentLabel, std::map<int, int>& labelReassignment,
                                            int& nextAvailableLabel) {
  auto labelMapping = labelReassignment.find(currentLabel);
  if (labelMapping == labelReassignment.end()) {
    int newLabel = nextAvailableLabel++;
    labelReassignment[currentLabel] = newLabel;
    return newLabel;
  }
  return labelMapping->second;
}

// Applies the labeling algorithm to the image, performing the labeling process for each pixel.
void karaseva_e_binaryimage_mpi::ApplyLabeling(std::vector<int>& inputImage, std::vector<int>& labeledImage, int rows,
                                               int cols, int startingLabel,
                                               std::map<int, std::set<int>>& label_connection_map) {
  int labelCounter = startingLabel;
  int dx[] = {-1, 0, -1};
  int dy[] = {0, -1, 1};

  for (int x = 0; x < rows; ++x) {
    for (int y = 0; y < cols; ++y) {
      int pos = x * cols + y;
      if (inputImage[pos] == 0 || labeledImage[pos] > 1) {
        HandlePixelLabeling(inputImage, labeledImage, label_connection_map, x, y, rows, cols, labelCounter, dx, dy);
      }
    }
  }

  FixLabelConnections(label_connection_map);

  // Final pass to assign root labels to each pixel.
  for (int x = 0; x < rows; ++x) {
    for (int y = 0; y < cols; ++y) {
      int pos = x * cols + y;
      if (labeledImage[pos] > 1) {
        int rootLabel = FindRootLabel(label_connection_map, labeledImage[pos]);
        labeledImage[pos] = rootLabel;
      }
    }
  }
}

// Handles the pixel labeling for each pixel in the image, considering its neighbors.
void karaseva_e_binaryimage_mpi::HandlePixelLabeling(std::vector<int>& inputImage, std::vector<int>& labeledImage,
                                                     std::map<int, std::set<int>>& label_connection_map, int x, int y,
                                                     int rows, int cols, int& labelCounter, int dx[], int dy[]) {
  int pos = x * cols + y;
  std::vector<int> neighboringLabels;

  // Check neighboring pixels for existing labels.
  for (int i = 0; i < 3; ++i) {
    int nx = x + dx[i];
    int ny = y + dy[i];
    int tmpPos = nx * cols + ny;
    if (nx >= 0 && nx < rows && ny >= 0 && ny < cols && labeledImage[tmpPos] > 1) {
      neighboringLabels.push_back(labeledImage[tmpPos]);
    }
  }

  if (neighboringLabels.empty() && labeledImage[pos] != 0) {  // If no neighbors, assign a new label.
    labeledImage[pos] = labelCounter++;
  } else {
    int minNeighborLabel = *std::min_element(neighboringLabels.begin(), neighboringLabels.end());
    labeledImage[pos] = minNeighborLabel;
    // Combine the labels of the neighbors.
    for (int label : neighboringLabels) {
      CombineLabels(label_connection_map, minNeighborLabel, label);
    }
  }
}

// Preprocessing step for the sequential task, which initializes the image data and labeled image.
bool karaseva_e_binaryimage_mpi::TestMPITaskSequential::PreProcessingImpl() {
  rows_ = task_data->inputs_count[0];
  columns_ = task_data->inputs_count[1];
  int totalPixels = rows_ * columns_;
  image_ = std::vector<int>(totalPixels);
  auto* inputData = reinterpret_cast<int*>(task_data->inputs[0]);
  std::copy(inputData, inputData + totalPixels, image_.begin());

  labeled_image_ = std::vector<int>(rows_ * columns_, 1);  // Initialize labeled image with all labels set to 1.
  return true;
}

// Validation step for the sequential task, which checks if the input image is a valid binary image.
bool karaseva_e_binaryimage_mpi::TestMPITaskSequential::ValidationImpl() {
  int tempRows = task_data->inputs_count[0];
  int tempColumns = task_data->inputs_count[1];
  auto* inputData = reinterpret_cast<int*>(task_data->inputs[0]);

  // Ensure all pixels are either 0 or 1.
  for (int x = 0; x < tempRows; ++x) {
    for (int y = 0; y < tempColumns; ++y) {
      int pixel = inputData[x * tempColumns + y];
      if (pixel < 0 || pixel > 1) {
        return false;  // Invalid pixel value.
      }
    }
  }
  return tempRows > 0 && tempColumns > 0 && static_cast<int>(task_data->outputs_count[0]) == tempRows &&
         static_cast<int>(task_data->outputs_count[1]) == tempColumns;
}

// Main implementation of the sequential task, which applies the labeling algorithm and corrects labels.
bool karaseva_e_binaryimage_mpi::TestMPITaskSequential::RunImpl() {
  std::map<int, std::set<int>> dummyMap;
  ApplyLabeling(image_, labeled_image_, rows_, columns_, 2, dummyMap);
  CorrectLabels(labeled_image_, rows_, columns_);
  return true;
}

// Post-processing step for the sequential task, which outputs the labeled image.
bool karaseva_e_binaryimage_mpi::TestMPITaskSequential::PostProcessingImpl() {
  auto* outputData = reinterpret_cast<int*>(task_data->outputs[0]);
  std::copy(labeled_image_.begin(), labeled_image_.end(), outputData);
  return true;
}

// Preprocessing for parallel task
bool karaseva_e_binaryimage_mpi::TestMPITaskParallel::PreProcessingImpl() {
  if (world_.rank() == 0) {
    rows_ = task_data->inputs_count[0];
    columns_ = task_data->inputs_count[1];
    int totalPixels = rows_ * columns_;
    image_ = std::vector<int>(totalPixels);

    auto* inputData = reinterpret_cast<int*>(task_data->inputs[0]);
    std::copy(inputData, inputData + totalPixels, image_.begin());

    // Initialize labeled_image_ with all pixels labeled as background (1)
    labeled_image_ = std::vector<int>(rows_ * columns_, 1);
  }

  std::cout << "Rank " << world_.rank() << " - PreProcessingImpl completed" << std::endl;

  return true;
}

// Validation for parallel task
bool karaseva_e_binaryimage_mpi::TestMPITaskParallel::ValidationImpl() {
  if (world_.rank() == 0) {
    int tempRows = task_data->inputs_count[0];
    int tempColumns = task_data->inputs_count[1];
    auto* inputData = reinterpret_cast<int*>(task_data->inputs[0]);

    // Validate that all pixel values are either 0 or 1
    for (int x = 0; x < tempRows; ++x) {
      for (int y = 0; y < tempColumns; ++y) {
        int pixel = inputData[x * tempColumns + y];
        if (pixel < 0 || pixel > 1) {
          return false;
        }
      }
    }

    // Ensure valid image dimensions
    return tempRows > 0 && tempColumns > 0 && static_cast<int>(task_data->outputs_count[0]) == tempRows &&
           static_cast<int>(task_data->outputs_count[1]) == tempColumns;
  }
  return true;
}

// Save label set to a string stream (for serialization)
void karaseva_e_binaryimage_mpi::Savelabel_set(std::ostringstream& oss, const std::set<int>& labelSet) {
  oss << labelSet.size() << " ";
  for (const auto& item : labelSet) {
    oss << item << " ";  // Save each element
  }
}

// Load label set from a string stream (for deserialization)
void karaseva_e_binaryimage_mpi::LoadLabelSet(std::istringstream& iss, std::set<int>& labelSet) {
  size_t size;
  iss >> size;
  labelSet.clear();
  for (size_t i = 0; i < size; ++i) {
    int item;
    iss >> item;  // Read each element
    labelSet.insert(item);
  }
}

// Custom serialization for std::map (map of labels and their connected labels)
void karaseva_e_binaryimage_mpi::SerializeLabelMap(std::ostringstream& oss,
                                                   const std::map<int, std::set<int>>& labelMap) {
  oss << labelMap.size() << " ";
  for (const auto& entry : labelMap) {
    oss << entry.first << " ";  // Save label
    Savelabel_set(oss, entry.second);
  }
}

// Custom deserialization for std::map (map of labels and their connected labels)
void karaseva_e_binaryimage_mpi::DeserializeLabelMap(std::istringstream& iss, std::map<int, std::set<int>>& labelMap) {
  size_t size;
  iss >> size;
  labelMap.clear();
  for (size_t i = 0; i < size; ++i) {
    int key;
    iss >> key;
    std::set<int> value;
    LoadLabelSet(iss, value);  // Deserialize the set
    labelMap[key] = value;
  }
}

// Main implementation of the parallel task
bool karaseva_e_binaryimage_mpi::TestMPITaskParallel::RunImpl() {
  std::cout << "Rank " << world_.rank() << " - RunImpl started" << std::endl;

  boost::mpi::broadcast(world_, rows_, 0);
  boost::mpi::broadcast(world_, columns_, 0);

  std::vector<int> partitionSizes(world_.size(), rows_ / world_.size() * columns_);
  for (int i = 0; i < rows_ % world_.size(); i++) {
    partitionSizes[i] += columns_;
  }

  local_image_ = std::vector<int>(partitionSizes[world_.rank()]);
  boost::mpi::scatterv(world_, image_, partitionSizes, local_image_.data(), 0);

  std::cout << "Rank " << world_.rank() << " - Image data scattered" << std::endl;

  // Perform local labeling on each process's partition
  std::vector<int> localLabeledImage(partitionSizes[world_.rank()], 1);
  int minLabel = 100000 * world_.rank() + 2;
  std::map<int, std::set<int>> localParentMap;
  ApplyLabeling(local_image_, localLabeledImage, partitionSizes[world_.rank()] / columns_, columns_, minLabel,
                localParentMap);

  boost::mpi::gatherv(world_, localLabeledImage, labeled_image_.data(), partitionSizes, 0);

  std::ostringstream oss;
  SerializeLabelMap(oss, localParentMap);
  std::string serializedData = oss.str();

  std::vector<int> dataSizes(world_.size());
  int dataSize = static_cast<int>(serializedData.size());
  boost::mpi::gather(world_, dataSize, dataSizes, 0);

  int bufferSize;
  std::vector<char> buffer;

  if (world_.rank() == 0) {
    bufferSize = std::accumulate(dataSizes.begin(), dataSizes.end(), 0);
    buffer = std::vector<char>(bufferSize);
  }

  std::vector<char> sendData(serializedData.begin(), serializedData.end());
  boost::mpi::gatherv(world_, sendData, buffer.data(), dataSizes, 0);

  std::cout << "Rank " << world_.rank() << " - Data gathered" << std::endl;

  if (world_.rank() == 0) {
    std::map<int, std::set<int>> globalMap;
    int displacement = 0;
    for (int i = 0; i < world_.size(); ++i) {
      std::string mapData = std::string(buffer.begin() + displacement, buffer.begin() + displacement + dataSizes[i]);
      std::istringstream inputStream(mapData);
      std::map<int, std::set<int>> receivedMap;
      DeserializeLabelMap(inputStream, receivedMap);
      displacement += dataSizes[i];
      globalMap.insert(receivedMap.begin(), receivedMap.end());
    }

    ApplyLabeling(image_, labeled_image_, rows_, columns_, 2, globalMap);
    CorrectLabels(labeled_image_, rows_, columns_);
  }

  std::cout << "Rank " << world_.rank() << " - RunImpl completed" << std::endl;

  return true;
}

// Post-processing for parallel task (copying the labeled image to output)
bool karaseva_e_binaryimage_mpi::TestMPITaskParallel::PostProcessingImpl() {
  boost::mpi::broadcast(world_, rows_, 0);
  boost::mpi::broadcast(world_, columns_, 0);

  auto* outputData = reinterpret_cast<int*>(task_data->outputs[0]);
  boost::mpi::gather(world_, labeled_image_.data(), rows_ * columns_, outputData, 0);

  return true;
}