class Value:
    """used to store scalar value and its gradient
       _children stores the elements child Values used to get
       to this value and _op stores the operation between them
    """
    def __init__(self, data, _children=(), _op=''):
        self.data = data
        self.grad = 0 # initializing grad as 0
        self._backward = lambda:None
        self._prev = set(_children)
        self._op = _op
        
    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other) # if other is not a Value, make it a Value
        out = Value(self.data+other.data, (self, other), '+')
        def _backward():
            self.grad += out.grad 
            other.grad += out.grad 
        out._backward = _backward
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data*other.data, (self, other), '*')
        def _backward():
            self.grad += other.data*out.grad # d(self)/d(out) = other.data
            other.grad += self.data*out.grad # d(other)/d(out) = self.data
        out._backward = _backward
        return out
    
    def __pow__(self, other):
        assert isinstance(other, (int, float))
        out = Value(self.data**other, (self,), f'**{other}')

        def _backward():
            self.grad += (other * self.data**(other-1)) * out.grad
        out._backward = _backward
        return out
    
    def relu(self):
        out = Value(0 if self.data < 0 else self.data, (self,), 'ReLU')
        def _backward():
            self.grad += (out.data > 0) * out.grad # d(self)/d(out) = 1 if out.data > 0 else 0
        out._backward = _backward
        return out
    
    def backward(self):
        # topological order all of the children in the graph
        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited: 
                visited.add(v) # mark as visited
                for child in v._prev: # visit all parents
                    build_topo(child) # recursive call to build topo
                topo.append(v) # add to topo list
        build_topo(self)
        
        self.grad = 1
        # go one variable at a time and apply the chain rule to get its gradient
        for v in reversed(topo):
            v._backward()

    def __neg__(self): # -self
        return self * -1

    def __radd__(self, other): # other + self
        return self + other

    def __sub__(self, other): # self - other
        return self + (-other)

    def __rsub__(self, other): # other - self
        return other + (-self)

    def __rmul__(self, other): # other * self
        return self * other

    def __truediv__(self, other): # self / other
        return self * other**-1

    def __rtruediv__(self, other): # other / self
        return other * self**-1

    def __repr__(self):
        return f"Value(data={self.data}, grad={self.grad})"