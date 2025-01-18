//
// Created by timetraveler314 on 1/18/25.
//

#ifndef OP_H
#define OP_H

class NdArray;
class Value;

class Op {
public:
    virtual ~Op() = default;

    virtual NdArray compute(std::vector<NdArray> args) = 0;
    virtual std::vector<NdArray> gradient(Value out_grad, std::vector<Value> args) = 0;
};



#endif //OP_H
