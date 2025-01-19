//
// Created by timetraveler314 on 1/19/25.
//

#ifndef GLOBAL_CUBLAS_HANDLE_CUH
#define GLOBAL_CUBLAS_HANDLE_CUH
#include <cublas_v2.h>

// Singleton class to manage a global cuBLAS handle
class global_cublas_handle {
    global_cublas_handle() {}

public:
    static cublasHandle_t& get_instance();

    global_cublas_handle(const global_cublas_handle&) = delete;
    global_cublas_handle& operator=(const global_cublas_handle&) = delete;
};

#endif //GLOBAL_CUBLAS_HANDLE_CUH
