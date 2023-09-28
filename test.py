import random
import numpy as np
import matplotlib.pyplot as plt
from pymicrograd.value import Value
from pymicrograd.nn import Neuron, Layer, MLP
np.random.seed(1337)
random.seed(1337)

from sklearn.datasets import make_moons, make_blobs
X, y = make_moons(n_samples=100, noise=0.1)

y = y*3 - 1

model = MLP(2, [16, 16, 1]) # 2-layer neural network
print(model)
print("number of parameters", len(model.parameters()))

def get_batch(X, y, batch_size=None):
    if batch_size is None: # if batch_size is None, use all data
        Xb, yb = X, y
    else: # else, use a random subset ("batch")
        ri = np.random.permutation(X.shape[0])[:batch_size] # random indices
        Xb, yb = X[ri], y[ri] # random subset
    inputs = [list(map(Value, xrow)) for xrow in Xb] # inputs to the network
    return inputs, yb

def loss(inputs, yb):
    # forward the model to get scores
    scores = list(map(model, inputs))
    
    # svm "max-margin" loss
    losses = [(1 + -yi*scorei).relu() for yi, scorei in zip(yb, scores)]
    data_loss = sum(losses) * (1.0 / len(losses))
    
    # L2 regularization
    alpha = 1e-4
    reg_loss = alpha * sum((p*p for p in model.parameters()))
    total_loss = data_loss + reg_loss
    
    return total_loss

# optimization
for k in range(100):
    # Get batch and calculate loss
    inputs, yb = get_batch(X, y)
    total_loss = loss(inputs, yb)
    print(total_loss)
    print(total_loss._prev)
    # Calculate accuracy
    scores = list(map(model, inputs))
    accuracy = [(yi > 0) == (scorei.data > 0) for yi, scorei in zip(yb, scores)]
    acc = sum(accuracy) / len(accuracy)
    
    # backward
    model.zero_grad()
    total_loss.backward()
    print(total_loss)
    print(total_loss._prev)
    print(list(total_loss._prev)[0]._prev)
    # update (sgd)
    learning_rate = 1.0 - 0.9*k/100
    for p in model.parameters():
        p.data -= learning_rate * p.grad
    
    if k % 1 == 0:
        print(f"step {k} loss {total_loss.data}, accuracy {acc*100}%")
