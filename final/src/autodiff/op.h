//
// Created by timetraveler314 on 1/18/25.
//

#ifndef OP_H
#define OP_H

class NdArray;
class ValueImpl;

using Value = std::shared_ptr<ValueImpl>;

class Op {
public:
    virtual ~Op() = default;

    virtual std::string name() const = 0;
    virtual NdArray compute(std::vector<NdArray>& args) const = 0;
    virtual std::vector<NdArray> gradient(Value out_grad, std::vector<Value>& args) const = 0;
};



#endif //OP_H
