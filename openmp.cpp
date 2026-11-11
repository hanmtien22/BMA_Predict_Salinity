#include <iostream>
#include <vector>
#include <omp.h> // Thư viện chứa các hàm Runtime của OpenMP

int main() {
    int n = 1000;
    std::vector<int> a(n, 1); // Mảng gồm 1000 phần tử, mỗi phần tử = 1
    long long total_sum = 0;

    // 1. Chỉ thị kết hợp Vùng song song + Chia sẻ công việc vòng lặp
    // - directive-name : parallel for
    // - clauses        : num_threads(4), reduction(+:total_sum)
    // - newline        : xuống dòng ngay sau chỉ thị
    #pragma omp parallel for num_threads(4) reduction(+:total_sum)
    for (int i = 0; i < n; i++) {
        total_sum += a[i];
        
        // Khối mã bên trong vòng lặp được phân chia cho 4 luồng thực thi
    }

    // 2. Sử dụng hàm thời gian chạy và in kết quả
    std::cout << "Tong tinh duoc: " << total_sum << std::endl;

    return 0;
}