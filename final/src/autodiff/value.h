//
// Created by timetraveler314 on 1/18/25.
//

#ifndef VALUE_H
#define VALUE_H

#include <memory>
#include <ndarray.h>
#include <optional>

#include "op.h"

class ValueImpl {
    std::shared_ptr<Op> op; // The operator that produced this value (optional, nullptr if this is a leaf)
    std::vector<Value> args; // The arguments to the operator
    bool requires_grad = false; // Whether this value requires gradient computation

    std::optional<NdArray> cached_data; // Cached data for this value

public:
    ValueImpl(std::shared_ptr<Op> op, const std::vector<Value> &args, const bool requires_grad, std::optional<NdArray> cached_data = std::nullopt)
        : op(std::move(op)), args(args), requires_grad(requires_grad) {}

    virtual ~ValueImpl() = default;

    // Create a leaf value
    explicit ValueImpl(const NdArray& data, const bool requires_grad = false): op(nullptr), requires_grad(requires_grad), cached_data(data) {}

    /*
     * realize() - Compute the value of this value,
     *             or return the cached value if it has already been computed
     *
     * @return The value of this value
     */
    NdArray realize() {
        if (!cached_data) {
            // Realize the arguments, then compute the value
            std::vector<NdArray> arg_data;
            for (const auto& arg : args) {
                arg_data.push_back(arg->realize());
            }
            cached_data = op->compute(arg_data);
        }
        return *cached_data;
    }

    bool isLeaf() const {
        return !op;
    }

    bool isRequiresGrad() const {
        return requires_grad;
    }

    auto& getArgs() {
        return args;
    }

    auto& getOp() {
        return op;
    }

    friend class Tensor;
};



#endif //VALUE_H
