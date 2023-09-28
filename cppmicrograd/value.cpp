#include <iostream>
#include <string>
#include <memory>
#include <set>

class Value {
public:
    Value(double data, std::set<std::shared_ptr<Value>> children = {}, const std::string& op = "")
        : data(data), grad(0), _prev(children), _op(op) {}

    std::shared_ptr<Value> operator+(std::shared_ptr<Value> other) {
        if (!other) other = std::make_shared<Value>(0);
        
        // Create a new set containing both the _prev set of this and the other object.
        std::set<std::shared_ptr<Value>> new_prev = _prev;
        new_prev.insert(other);
        
        auto out = std::shared_ptr<Value>(new Value(this->data + other->data, new_prev, "+"));
        return out;
    }

    double data;
    double grad;
private:
    std::set<std::shared_ptr<Value>> _prev;
    std::string _op;
};

int main() {
    auto val1 = std::make_shared<Value>(5);
    auto val2 = std::make_shared<Value>(3);
    auto sum = *val1 + val2;
    std::cout << sum->data << std::endl; // Output should be 8
    return 0;
}
