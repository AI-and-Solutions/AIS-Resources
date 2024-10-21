import torch
import torch.nn.functional as F
import math

# Assuming x_train and y_train are defined
# x_train: Input data (e.g., MNIST images), shape: (n_samples, 784)
# y_train: Target labels, shape: (n_samples,)

# Initialize weights and bias
weights = torch.randn(784, 10) / math.sqrt(784)
weights.requires_grad_()
bias = torch.zeros(10, requires_grad=True)

# Define the model
def model(xb):
    return xb @ weights + bias  # Linear transformation (no softmax needed)

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
        loss = F.cross_entropy(pred, yb)  # calculate loss using cross-entropy

        loss.backward()  # backpropagation
        with torch.no_grad():
            weights -= weights.grad * lr  # update weights
            bias -= bias.grad * lr  # update bias
            weights.grad.zero_()  # reset gradients
            bias.grad.zero_()  # reset gradients

# Check final loss and accuracy
final_loss = F.cross_entropy(model(xb), yb)  # using the updated model for prediction
final_accuracy = accuracy(model(xb), yb)

# Print the results
print(final_loss, final_accuracy)
