#include "seq/budazhapova_betcher_odd_even_merge_seq/include/odd_even_merge.hpp"

#include <algorithm>
#include <thread>

namespace budazhapova_betcher_odd_even_merge_seq {
    void CountingSort(std::vector<int>& arr, int exp) {
        int n = arr.size();
        std::vector<int> output(n);
        std::vector<int> count(10, 0);

        for (int i = 0; i < n; i++) {
            int index = (arr[i] / exp) % 10;
            count[index]++;
        }
        for (int i = 1; i < 10; i++) {
            count[i] += count[i - 1];
        }
        for (int i = n - 1; i >= 0; i--) {
            int index = (arr[i] / exp) % 10;
            output[count[index] - 1] = arr[i];
            count[index]--;
        }
        for (int i = 0; i < n; i++) {
            arr[i] = output[i];
        }
    }

    void RadixSort(std::vector<int>& arr) {
        int max_num = *std::max_element(arr.begin(), arr.end());
        for (int exp = 1; max_num / exp > 0; exp *= 10) {
            CountingSort(arr, exp);
        }
    }
}  // namespace budazhapova_betcher_odd_even_merge_seq
bool budazhapova_betcher_odd_even_merge_seq::MergeSequential::PreProcessing() {

    res_ = std::vector<int>(reinterpret_cast<int*>(task_data->inputs[0]),
        reinterpret_cast<int*>(task_data->inputs[0]) + task_data->inputs_count[0]);
    n_el_ = task_data->inputs_count[0];
    return true;
}

bool budazhapova_betcher_odd_even_merge_seq::MergeSequential::Validation() {

    return task_data->inputs_count[0] > 0;
}

bool budazhapova_betcher_odd_even_merge_seq::MergeSequential::Run() {
    RadixSort(res_);
    return true;
}

bool budazhapova_betcher_odd_even_merge_seq::MergeSequential::PostProcessing() {
    
    int* output = reinterpret_cast<int*>(task_data->outputs[0]);
    for (size_t i = 0; i < res_.size(); i++) {
        output[i] = res_[i];
    }
    return true;
}