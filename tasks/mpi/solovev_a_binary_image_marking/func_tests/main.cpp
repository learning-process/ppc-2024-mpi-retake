#include <gtest/gtest.h>

#include <boost/mpi/collectives.hpp>
#include <boost/mpi/communicator.hpp>
#include <cstdint>
#include <memory>
#include <random>
#include <vector>

#include "core/task/include/task.hpp"
#include "mpi/solovev_a_binary_image_marking/include/ops_mpi.hpp"

namespace {
std::vector<int> randomImg(int height, int width) {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<> dis(0, 1);

  std::vector<int> img(height * width);

  for (int& pixel : img) {
    pixel = dis(gen);
  }

  return img;
}

void TestBodyFunction(int height, int width) {
  boost::mpi::communicator world;

  std::vector<int> resultMPI(height * width);
  std::vector<int> resultSEQ(height * width);

  std::vector<int> expected_result(height * width, 0);

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();

  std::vector<int> img = randomImg(height, width);
  taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(const_cast<int*>(&height)));
  taskDataPar->inputs_count.emplace_back(1);
  taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(const_cast<int*>(&width)));
  taskDataPar->inputs_count.emplace_back(1);
  taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(const_cast<int*>(img.data())));
  taskDataPar->inputs_count.emplace_back(img.size());

  taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(resultMPI.data()));
  taskDataPar->outputs_count.emplace_back(resultMPI.size());

  if (world.rank() == 0) {
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(const_cast<int*>(&height)));
    taskDataSeq->inputs_count.emplace_back(1);
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(const_cast<int*>(&width)));
    taskDataSeq->inputs_count.emplace_back(1);
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(const_cast<int*>(img.data())));
    taskDataSeq->inputs_count.emplace_back(img.size());

    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(resultSEQ.data()));
    taskDataSeq->outputs_count.emplace_back(resultSEQ.size());

    solovev_a_binary_image_marking::TestMPITaskSequential BinaryMarkerSeq(taskDataSeq);
    ASSERT_EQ(BinaryMarkerSeq.ValidationImpl(), true);
    BinaryMarkerSeq.PreProcessingImpl();
    BinaryMarkerSeq.RunImpl();
    BinaryMarkerSeq.PostProcessingImpl();

    expected_result = std::move(resultSEQ);
  }

  solovev_a_binary_image_marking::TestMPITaskParallel BinaryMarkerMPI(taskDataPar);
  ASSERT_EQ(BinaryMarkerMPI.ValidationImpl(), true);
  BinaryMarkerMPI.PreProcessingImpl();
  BinaryMarkerMPI.RunImpl();
  BinaryMarkerMPI.PostProcessingImpl();

  if (world.rank() == 0) {
    for (size_t i = 0; i < resultMPI.size(); ++i) {
      ASSERT_EQ(resultMPI[i], expected_result[i]);
    }
  }
}

void ValidationFalseTest(int height, int width) {
  boost::mpi::communicator world;

  std::vector<int> img{};
  std::vector<int> resultMPI;

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(const_cast<int*>(&height)));
    taskDataPar->inputs_count.emplace_back(1);
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(const_cast<int*>(&width)));
    taskDataPar->inputs_count.emplace_back(1);
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(const_cast<int*>(img.data())));
    taskDataPar->inputs_count.emplace_back(img.size());

    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(resultMPI.data()));
    taskDataPar->outputs_count.emplace_back(resultMPI.size());
  }

  solovev_a_binary_image_marking::TestMPITaskParallel BinaryMarkerMPI(taskDataPar);
  if (world.rank() == 0) {
    ASSERT_EQ(BinaryMarkerMPI.ValidationImpl(), false);
  }
}

}  // namespace

TEST(solovev_a_binary_image_marking, Test_image_random_5X5) { TestBodyFunction(5, 5); }

TEST(solovev_a_binary_image_marking, Test_image_random_11X11) { TestBodyFunction(11, 11); }

TEST(solovev_a_binary_image_marking, Test_image_random_16X16) { TestBodyFunction(16, 16); }

TEST(solovev_a_binary_image_marking, Test_image_random_32X32) { TestBodyFunction(32, 32); }

TEST(solovev_a_binary_image_marking, Test_image_random_23X31) { TestBodyFunction(23, 31); }

TEST(solovev_a_binary_image_marking, Test_image_random_31X23) { TestBodyFunction(31, 23); }

TEST(solovev_a_binary_image_marking, Test_image_random_50X50) { TestBodyFunction(50, 50); }

TEST(solovev_a_binary_image_marking, Test_image_random_75X75) { TestBodyFunction(75, 75); }

TEST(solovev_a_binary_image_marking, Whole_image) {
  boost::mpi::communicator world;

  int height = 77;
  int width = 77;

  std::vector<int> img(height * width, 1);
  std::vector<int> resultMPI(height * width);
  std::vector<int> resultSEQ(height * width);

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(const_cast<int*>(&height)));
    taskDataPar->inputs_count.emplace_back(1);
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(const_cast<int*>(&width)));
    taskDataPar->inputs_count.emplace_back(1);
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(const_cast<int*>(img.data())));
    taskDataPar->inputs_count.emplace_back(img.size());

    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(resultMPI.data()));
    taskDataPar->outputs_count.emplace_back(resultMPI.size());

    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(const_cast<int*>(&height)));
    taskDataSeq->inputs_count.emplace_back(1);
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(const_cast<int*>(&width)));
    taskDataSeq->inputs_count.emplace_back(1);
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(const_cast<int*>(img.data())));
    taskDataSeq->inputs_count.emplace_back(img.size());

    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(resultSEQ.data()));
    taskDataSeq->outputs_count.emplace_back(resultSEQ.size());
  }

  solovev_a_binary_image_marking::TestMPITaskParallel BinaryMarkerMPI(taskDataPar);
  ASSERT_EQ(BinaryMarkerMPI.ValidationImpl(), true);
  BinaryMarkerMPI.PreProcessingImpl();
  BinaryMarkerMPI.RunImpl();
  BinaryMarkerMPI.PostProcessingImpl();

  if (world.rank() == 0) {
    solovev_a_binary_image_marking::TestMPITaskSequential BinaryMarkerSeq(taskDataSeq);
    ASSERT_EQ(BinaryMarkerSeq.ValidationImpl(), true);
    BinaryMarkerSeq.PreProcessingImpl();
    BinaryMarkerSeq.RunImpl();
    BinaryMarkerSeq.PostProcessingImpl();

    for (size_t i = 0; i < resultMPI.size(); ++i) {
      ASSERT_EQ(resultMPI[i], resultSEQ[i]);
    }
  }
}

TEST(solovev_a_binary_image_marking, Validation_false_1) { ValidationFalseTest(-1, 10); }

TEST(solovev_a_binary_image_marking, Validation_false_2) { ValidationFalseTest(10, -1); }

TEST(solovev_a_binary_image_marking, Validation_false_3) { ValidationFalseTest(10, 10); }