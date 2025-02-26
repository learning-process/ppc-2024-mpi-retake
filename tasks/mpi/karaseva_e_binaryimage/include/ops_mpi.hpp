#pragma once

#include <boost/mpi/collectives.hpp>
#include <boost/mpi/communicator.hpp>
#include <iostream>
#include <map>
#include <numeric>
#include <set>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

#include "core/task/include/task.hpp"

namespace karaseva_e_binaryimage_mpi {

// Forward declarations of functions

// Label management functions
int findRootLabel(std::map<int, std::set<int>>& labelConnectionMap, int label);
void combineLabels(std::map<int, std::set<int>>& labelConnectionMap, int label1, int label2);
void fixLabelConnections(std::map<int, std::set<int>>& labelConnectionMap);
void correctLabels(std::vector<int>& labeledImage, int rows, int cols);
int assignLabel(int currentLabel, std::map<int, int>& labelReassignment, int& nextAvailableLabel);

// Labeling functions
void applyLabeling(std::vector<int>& inputImage, std::vector<int>& labeledImage, int rows, int cols, int startingLabel,
                   std::map<int, std::set<int>>& labelConnectionMap);
void handlePixelLabeling(std::vector<int>& inputImage, std::vector<int>& labeledImage,
                         std::map<int, std::set<int>>& labelConnectionMap, int x, int y, int rows, int cols,
                         int& labelCounter, int dx[], int dy[]);

// Label connection functions
void createNewLabelConnection(std::map<int, std::set<int>>& labelConnectionMap, int label1, int label2);
void connectWithExistingLabel(std::map<int, std::set<int>>& labelConnectionMap, int existingLabel, int newLabel);
void mergeLabelConnections(std::map<int, std::set<int>>& labelConnectionMap, int label1, int label2);

// Serialization functions
void saveLabelSet(std::ostringstream& oss, const std::set<int>& labelSet);
void loadLabelSet(std::istringstream& iss, std::set<int>& labelSet);
void serializeLabelMap(std::ostringstream& oss, const std::map<int, std::set<int>>& labelMap);
void deserializeLabelMap(std::istringstream& iss, std::map<int, std::set<int>>& labelMap);

// TestMPITaskSequential class declaration
class TestMPITaskSequential : public ppc::core::Task {
 public:
  explicit TestMPITaskSequential(ppc::core::TaskDataPtr task_data) : Task(std::move(task_data)) {}

  bool PreProcessingImpl() override;
  bool ValidationImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

 private:
  int rows_;
  int columns_;
  std::vector<int> image_;
  std::vector<int> labeled_image;
};

// TestMPITaskParallel class declaration
class TestMPITaskParallel : public ppc::core::Task {
 public:
  explicit TestMPITaskParallel(ppc::core::TaskDataPtr task_data) : Task(std::move(task_data)) {}

  bool PreProcessingImpl() override;
  bool ValidationImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

 private:
  boost::mpi::communicator world;
  int rows_;
  int columns_;
  std::vector<int> image_;
  std::vector<int> labeled_image;
  std::vector<int> local_image_;
};

}  // namespace karaseva_e_binaryimage_mpi