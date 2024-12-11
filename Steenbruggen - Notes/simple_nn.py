import torch
import math

# Assuming x_train and y_train are defined
# x_train: Input data (e.g., MNIST images), shape: (n_samples, 784)
# y_train: Target labels, shape: (n_samples,)

# Initialize weights and bias
weights = torch.randn(784, 10) / math.sqrt(784)
weights.requires_grad_()
bias = torch.zeros(10, requires_grad=True)

# Define log_softmax activation function
def log_softmax(x):
    return x - x.exp().sum(dim=-1, keepdim=True).log()

# Define the model
def model(xb):
    return log_softmax(xb @ weights + bias)

# Define negative log likelihood loss function
def nll(input, target):
    return -input[range(target.shape[0]), target].mean()

# Define accuracy calculation
def accuracy(out, yb):
    preds = torch.argmax(out, dim=1)
    return (preds == yb).float().mean()

# Training parameters
bs = 64  # batch size
lr = 0.5  # learning rate
epochs = 2  # number of epochs

# Training loop
n = x_train.shape[0]  # total number of samples

for epoch in range(epochs):
    for i in range((n - 1) // bs + 1):
        start_i = i * bs
        end_i = start_i + bs
        xb = x_train[start_i:end_i]  # mini-batch inputs
        yb = y_train[start_i:end_i]  # mini-batch targets
        
        pred = model(xb)  # forward pass
        loss = nll(pred, yb)  # calculate loss

        loss.backward()  # backpropagation
        with torch.no_grad():
            weights -= weights.grad * lr  # update weights
            bias -= bias.grad * lr  # update bias
            weights.grad.zero_()  # reset gradients
            bias.grad.zero_()  # reset gradients

# Check final loss and accuracy
print(loss_func(model(xb), yb), accuracy(model(xb), yb))
