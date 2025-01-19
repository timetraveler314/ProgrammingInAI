//
// Created by timetraveler314 on 1/19/25.
//

#include "global_cublas_handle.cuh"

#include <cublas_v2.h>

cublasHandle_t &global_cublas_handle::get_instance() {
    static cublasHandle_t instance;
    static bool initialized = false;

    if (!initialized) {
        cublasCreate(&instance);
        initialized = true;
    }

    return instance;
}
