#pragma once

#include <boost/mpi/collectives.hpp>
#include <boost/mpi/communicator.hpp>
#include <map>
#include <set>
#include <sstream>
#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

namespace karaseva_e_binaryimage_mpi {

// Function declaration
std::vector<int> CreateRandomBinaryImage(int r, int c);

// Label management functions
int FindRootLabel(std::map<int, std::set<int>>& label_connection_map, int label);
void CombineLabels(std::map<int, std::set<int>>& label_connection_map, int label1, int label2);
void FixLabelConnections(std::map<int, std::set<int>>& label_connection_map);
void CorrectLabels(std::vector<int>& labeled_image, int rows, int cols);
int AssignLabel(int current_label, std::map<int, int>& label_reassignment, int& next_available_label);

// Labeling functions
void ApplyLabeling(std::vector<int>& input_image, std::vector<int>& labeled_image, int rows, int cols,
                   int starting_label, std::map<int, std::set<int>>& label_connection_map);
void HandlePixelLabeling(std::vector<int>& input_image, std::vector<int>& labeled_image,
                         std::map<int, std::set<int>>& label_connection_map, int x, int y, int rows, int cols,
                         int& label_counter, const int dx[], const int dy[]);

// Label connection functions
void CreatenewLabelConnection(std::map<int, std::set<int>>& label_connection_map, int label1, int label2);
void ConnectWithexistingLabel(std::map<int, std::set<int>>& label_connection_map, int existing_label, int new_label);
void MergeLabelConnections(std::map<int, std::set<int>>& label_connection_map, int label1, int label2);

// Serialization functions
void SavelabelSet(std::ostringstream& oss, const std::set<int>& label_set);
void LoadLabelSet(std::istringstream& iss, std::set<int>& label_set);
void SerializelabelMap(std::ostringstream& oss, const std::map<int, std::set<int>>& label_map);
void DeserializelabelMap(std::istringstream& iss, std::map<int, std::set<int>>& label_map);

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
  std::vector<int> labeled_image_;
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
  boost::mpi::communicator world_;
  int rows_;
  int columns_;
  std::vector<int> image_;
  std::vector<int> labeled_image_;
  std::vector<int> local_image_;
};

}  // namespace karaseva_e_binaryimage_mpi