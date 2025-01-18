//
// Created by timetraveler314 on 1/18/25.
//

#ifndef VALUE_H
#define VALUE_H

#include <optional>

#include "op.h"

class Value {
    std::optional<Op> op; // The operator that produced this value
    std::vector<Value> args; // The arguments to the operator

};



#endif //VALUE_H
