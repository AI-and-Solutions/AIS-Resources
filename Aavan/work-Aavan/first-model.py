import math
import torch  # Add this import

# Initialize weights and bias
weights = torch.randn(784, 10) / math.sqrt(784)
weights.requires_grad_()
bias = torch.zeros(10, requires_grad=True)

# Log-Softmax function
def log_softmax(x):
    return x - x.exp().sum(-1).log().unsqueeze(-1)

# Model function
def model(xb):
    return log_softmax(xb @ weights + bias)

# Batch size
bs = 64

# Mini-batch from x_train
xb = x_train[0:bs]  

# Predictions
preds = model(xb)
print(preds[0], preds.shape)

# Negative Log-Likelihood Loss
def nll(input, target):
    return -input[range(target.shape[0]), target].mean()

loss_func = nll

# Mini-batch from y_train
yb = y_train[0:bs]  

# Calculate and print loss
print(loss_func(preds, yb))

# Accuracy calculation
def accuracy(out, yb):
    preds = torch.argmax(out, dim=1)
    return (preds == yb).float().mean()

# Print accuracy
print(accuracy(preds, yb))
#does not run yet, idk why