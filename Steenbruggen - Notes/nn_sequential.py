import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Custom Lambda layer to apply a function
class Lambda(nn.Module):
    def __init__(self, func):
        super().__init__()
        self.func = func

    def forward(self, x):
        return self.func(x)

# Function to preprocess data
def preprocess(x):
    return x.view(-1, 1, 28, 28)  # Reshape to (batch_size, 1, height, width)

# Example datasets (replace these with your actual datasets)
x_train = torch.randn(1000, 28, 28)  # 1000 images of 28x28
y_train = torch.randint(0, 10, (1000,))  # 1000 labels for 10 classes

x_valid = torch.randn(200, 28, 28)  # 200 validation images
y_valid = torch.randint(0, 10, (200,))  # 200 validation labels

# Create TensorDataset and DataLoader
train_ds = TensorDataset(x_train, y_train)
valid_ds = TensorDataset(x_valid, y_valid)

def get_data(train_ds, valid_ds, bs):
    train_dl = DataLoader(train_ds, batch_size=bs, shuffle=True)
    valid_dl = DataLoader(valid_ds, batch_size=bs)
    return train_dl, valid_dl

train_dl, valid_dl = get_data(train_ds, valid_ds, bs=64)

# Define the CNN model using nn.Sequential
model = nn.Sequential(
    Lambda(preprocess),  # Custom layer to preprocess input
    nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1),
    nn.ReLU(),
    nn.Conv2d(16, 16, kernel_size=3, stride=2, padding=1),
    nn.ReLU(),
    nn.Conv2d(16, 10, kernel_size=3, stride=2, padding=1),
    nn.ReLU(),
    nn.AdaptiveAvgPool2d(1),  # Adaptive pooling to output size 1x1
    Lambda(lambda x: x.view(x.size(0), -1)),  # Flatten the output
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
