#include <string>
#include <vector>

#include "core/task/include/task.hpp"

namespace fomin_v_sentence_count {

class SentenceCountSequential : public ppc::core::Task {
 public:
  explicit SentenceCountSequential(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}
  bool PreProcessingImpl() override;
  bool ValidationImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

 private:
  std::string input_;
  int sentence_count;
};

}  // namespace fomin_v_sentence_count