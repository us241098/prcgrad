import random
from typing import Any
from pymicrograd.value import Value

class Module:
    # Base class for all neural network modules
    def zero_grad(self):
        # zero out the gradients of all parameters before the backward pass
        for p in self.parameters():
            p.grad = 0
            
    def parameters(self):
        # get all parameters
        return []
    
class Neuron(Module):
    def __init__(self, nin, nonlin=True):
        self.w = [Value(random.uniform(-1,1)) for _ in range(nin)] # random weights for each input nin is the number of inputs
        self.b = Value(0)
        self.nonlin = nonlin # if nonlin is true, apply relu nonlinearity
    
    def __call__(self, x):
        # forward pass
        act = sum((wi*xi for wi,xi in zip(self.w, x)), self.b) # multiply each weight by each input and sum them up and add the bias
        return act.relu() if self.nonlin else act # if nonlin is true, apply relu nonlinearity
    
    def parameters(self):
        return self.w + [self.b] # return all parameters
    
    def __repr__(self):
        return f"{'ReLU' if self.nonlin else 'Linear'}Neuron({len(self.w)})"
    

class Layer(Module):
    def __init__(self, nin, nout, **kwargs):
        self.neurons = [Neuron(nin, **kwargs) for _ in range(nout)] # create nout neurons in this layer
        
    def __call__(self, x):
        out = [n(x) for n in self.neurons]
        return out[0] if len(out) == 1 else out
    
    def parameters(self):
        return [p for n in self.neurons for p in n.parameters()] # return all parameters for all neurons in this layer
    
    def __repr__(self):
        return f"Layer of [{', '.join(str(n) for n in self.neurons)}]"
    
class MLP(Module):
    def __init__(self, nin, nouts):
        sz = [nin] + nouts
        self.layers = [Layer(sz[i], sz[i+1], nonlin=i!=len(nouts)-1) for i in range(len(nouts))]
    
    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
    
    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]
    
    def __repr__(self):
        return f"MLP of [{', '.join(str(layer) for layer in self.layers)}]"