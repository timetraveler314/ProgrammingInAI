//
// Created by timetraveler314 on 1/19/25.
//

#include <set>
#include <map>
#include "autodiff.h"

static std::vector<Value> find_topological_order(const std::vector<Value> &values) {
    std::set<Value> visited;
    std::vector<Value> order;
    for (const auto &value: values) {
        if (!visited.contains(value)) {
            std::function<void(const Value &)> dfs = [&](const Value &v) {
                visited.insert(v);
                for (const auto &child: v->getArgs()) {
                    if (!visited.contains(child)) {
                        dfs(child);
                    }
                }
                order.push_back(v);
            };
            dfs(value);
        }
    }

    return order;
}

void compute_gradients(const Tensor &out_tensor, const Tensor &out_grad) {
    std::map<Tensor, std::vector<NdArray>> node_to_grads;
    node_to_grads[out_tensor] = {out_grad.to_value()->realize()};

    auto topological_order = find_topological_order({out_tensor.to_value()});

    std::stringstream ss;
    ss << "=== AutoDiff Topological Order ===" << std::endl;
    for (const auto &node: topological_order) {
        if (node->getOp()) {
            ss << node->getOp()->name() << " -> " << std::endl;
        } else {
            ss << "<leaf> -> " << std::endl;
        }
    }

    // std::cout << ss.str() << std::endl;

    // In reverse topological order
    for (auto it = topological_order.rbegin(); it != topological_order.rend(); ++it) {
        auto node = *it;
        auto tensor_node = Tensor(std::dynamic_pointer_cast<TensorImpl>(node));
        auto &grads = node_to_grads[tensor_node];

        // Compute the gradient of the node
        NdArray current_grad = grads[0];
        for (int i = 1; i < grads.size(); i++) {
            current_grad = current_grad + grads[i];
        }

        // Set the gradient of the node
        if (node->isRequiresGrad()) {
            tensor_node.setGrad(current_grad);
        }

        if (node->isLeaf()) continue;

        // For compute node, back-propagate the gradient to the arguments
        auto grad = node->getOp()->gradient(current_grad, node->getArgs());
        // For-loop of zip(grad, node->getArgs())
        for (int i = 0; i < node->getArgs().size(); i++) {
            auto arg = Tensor(std::dynamic_pointer_cast<TensorImpl>(node->getArgs()[i]));
            node_to_grads[arg].push_back(grad[i]);
        }
    }
}
