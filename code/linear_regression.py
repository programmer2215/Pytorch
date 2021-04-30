import torch 
import numpy as np


# enter the parameters
# in format [<temp>, <rainfall>, <humidity>]
# each row is for different regions
inputs = np.array([
    [73, 67, 43],
    [91, 88, 64],
    [87, 134, 58],
    [102, 43, 37],
    [69, 96, 70]],
    dtype = 'float32'
)


# enter the target values
# in format [<apple yield>, <orange yield>]
# each row is for different regions
targets = np.array([
    [56, 70],
    [81, 101],
    [119, 133],
    [22, 37],
    [103, 119]],
    dtype = 'float32'
)


# convert numpy array to tensor
inputs = torch.from_numpy(inputs)
targets = torch.from_numpy(targets)


# generate random weights and biases

w = torch.randn(2, 3, requires_grad=True)
b = torch.randn(2, requires_grad=True)
print("WEIGHTS: ")
print(w, "\n")
print("BIASES: ")
print(b, "\n")

# now create a model
def model(x):
    return x @ w.t() + b



# create Loss Function (Mean Square Error is used)

def mse(t1, t2):
    diff = t1 - t2
    return torch.sum(diff * diff) / diff.numel()



# now using Gradient descent reduce the Loss





preds = model(inputs)
loss = mse(preds, targets)
print("Reduced Loss: ")
print(loss)
for epoch in range(1, 10000):
    preds = model(inputs)
    loss = mse(preds, targets)
    loss.backward()
    with torch.no_grad():
        w -= w.grad * 1e-4
        b -= b.grad * 1e-4
        w.grad.zero_()
        b.grad.zero_()
    print(f"Loss: {loss} [EPOCH: {epoch}/100]")
print("TARGET: ",targets)
print("preds: ",preds)
