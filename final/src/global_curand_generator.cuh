//
// Created by timetraveler314 on 10/28/24.
//
// A singleton class to manage a global curand generator

#ifndef GLOBAL_CURAND_GENERATOR_CUH
#define GLOBAL_CURAND_GENERATOR_CUH
#include <curand.h>


class global_curand_generator {
    global_curand_generator() {}

public:
    static curandGenerator_t& get_instance(unsigned long long seed = 42ULL) {
        static curandGenerator_t instance;
        static bool initialized = false;

        if (!initialized) {
            curandCreateGenerator(&instance, CURAND_RNG_PSEUDO_DEFAULT);
            curandSetPseudoRandomGeneratorSeed(instance, seed);
            initialized = true;
        }

        return instance;
    }

    global_curand_generator(const global_curand_generator&) = delete;
    global_curand_generator& operator=(const global_curand_generator&) = delete;
};



#endif //GLOBAL_CURAND_GENERATOR_CUH
