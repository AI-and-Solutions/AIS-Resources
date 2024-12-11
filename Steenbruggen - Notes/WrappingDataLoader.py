import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Function to preprocess data
def preprocess(x, y):
    return x.view(-1, 1, x.size(1), x.size(2)), y  # Reshape to (batch_size, 1, height, width)

# Wrapped DataLoader class
class WrappedDataLoader:
    def __init__(self, dl, func):
        self.dl = dl
        self.func = func

    def __len__(self):
        return len(self.dl)

    def __iter__(self):
        for b in self.dl:
            yield self.func(*b)

# Function to create DataLoader
def get_data(train_ds, valid_ds, bs):
    train_dl = DataLoader(train_ds, batch_size=bs, shuffle=True)
    valid_dl = DataLoader(valid_ds, batch_size=bs)
    return train_dl, valid_dl

# Example datasets (replace these with your actual datasets)
# Let's say x_train and y_train are your training images and labels.
# For demonstration, let's create random tensors.
# Make sure to replace this with your actual image dataset.
x_train = torch.randn(1000, 28, 28)  # 1000 images of 28x28
y_train = torch.randint(0, 10, (1000,))  # 1000 labels for 10 classes

x_valid = torch.randn(200, 28, 28)  # 200 validation images
y_valid = torch.randint(0, 10, (200,))  # 200 validation labels

# Create TensorDataset and DataLoader
train_ds = TensorDataset(x_train, y_train)
valid_ds = TensorDataset(x_valid, y_valid)
train_dl, valid_dl = get_data(train_ds, valid_ds, bs=64)

# Wrap DataLoaders
train_dl = WrappedDataLoader(train_dl, preprocess)
valid_dl = WrappedDataLoader(valid_dl, preprocess)

# Define the CNN model
model = nn.Sequential(
    nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1),
    nn.ReLU(),
    nn.Conv2d(16, 16, kernel_size=3, stride=2, padding=1),
    nn.ReLU(),
    nn.Conv2d(16, 10, kernel_size=3, stride=2, padding=1),
    nn.ReLU(),
    nn.AdaptiveAvgPool2d(1),  # Adaptive pooling to output size 1x1
    nn.Flatten()  # Flatten the output
)

# Loss function and optimizer
loss_func = nn.CrossEntropyLoss()
opt = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

# Training function
def fit(epochs, model, loss_func, opt, train_dl, valid_dl):
    for epoch in range(epochs):
        model.train()
        for xb, yb in train_dl:
            pred = model(xb)  # Forward pass
            loss = loss_func(pred, yb)  # Calculate loss
            opt.zero_grad()  # Reset gradients
            loss.backward()  # Backpropagation
            opt.step()  # Update parameters

        # Validation
        model.eval()
        with torch.no_grad():
            val_loss = sum(loss_func(model(xb), yb) for xb, yb in valid_dl) / len(valid_dl)
        print(f'Epoch {epoch}: Training loss: {loss.item()}, Validation loss: {val_loss.item()}')

# Train the model
fit(epochs=5, model=model, loss_func=loss_func, opt=opt, train_dl=train_dl, valid_dl=valid_dl)
