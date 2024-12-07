import torch
import torch.nn as nn
import torch.optim as optim
from loadData import train_loader, val_loader  # Import the data loaders

class TicTacToeNN(nn.Module):
    def __init__(self):
        super(TicTacToeNN, self).__init__()
        self.fc1 = nn.Linear(9, 16)  # Input layer to first hidden layer
        self.fc2 = nn.Linear(16, 16)  # First hidden layer to second hidden layer
        self.fc3 = nn.Linear(16, 9)  # Second hidden layer to output layer
        self.relu = nn.ReLU()  # Activation function
        self.softmax = nn.Softmax(dim=1)  # Softmax for output layer

    def forward(self, x):
        x = self.relu(self.fc1(x))  # Input to hidden layer 1
        x = self.relu(self.fc2(x))  # Hidden layer 1 to hidden layer 2
        x = self.softmax(self.fc3(x))  # Hidden layer 2 to output layer
        return x

# Initialize the model
model = TicTacToeNN()

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()  # Suitable for classification tasks
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Add scheduler for dynamic learning rate
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

# Enhanced training loop with validation accuracy tracking
def train(model, optimizer, criterion, epochs, train_loader, val_loader):
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for board_states, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(board_states)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        scheduler.step()  # Update learning rate

        # Validation phase
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for board_states, labels in val_loader:
                outputs = model(board_states)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        val_accuracy = 100 * correct / total
        print(f"Epoch {epoch+1}, Loss: {running_loss / len(train_loader):.4f}, Validation Accuracy: {val_accuracy:.2f}%")

        if val_accuracy >= 90.0:
            print("Target reached!")
            break

        # Save the trained model
    torch.save(model.state_dict(), "model.pth")
    print("Model saved as model.pth")

# Train the model
train(model, optimizer, criterion, epochs=20, train_loader=train_loader, val_loader=val_loader)
