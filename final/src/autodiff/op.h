//
// Created by timetraveler314 on 1/18/25.
//

#ifndef OP_H
#define OP_H

class NdArray;
class ValueImpl;

/* Value - represents a node in the computation graph
 * all values must implement the realize method that computes the value
 * of the node in the graph
 *
 * To support polymorphism, we use a shared pointer to the real implementation
 */
using Value = std::shared_ptr<ValueImpl>;

/* Op - base class for all operators
 * all operators must implement the compute and gradient methods
 * that state how forward and backward passes are computed
 */
class Op {
public:
    virtual ~Op() = default;

    virtual std::string name() const = 0;
    virtual NdArray compute(std::vector<NdArray>& args) const = 0;
    virtual std::vector<NdArray> gradient(const NdArray out_grad, std::vector<Value>& args) const = 0;
};



#endif //OP_H
