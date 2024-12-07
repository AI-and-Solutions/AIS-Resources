## The Blurb

The Tic-Tac-Toe model is designed to determine the optimal move for any given valid Tic-Tac-Toe board. It is implemented in Python using the PyTorch library. To train the model, we use a program that generates various board states along with their corresponding optimal moves.


### README for Tic-Tac-Toe Model Training with PyTorch

This is for files Data.py, loadData.py, and NNTicTacToe.py

This project aims to train a neural network to predict the optimal move for a given Tic-Tac-Toe board using PyTorch. The dataset consists of various board configurations and their optimal moves, generated using the earlier Tic-Tac-Toe data generation code. The neural network model is trained to predict the best move based on the current board state.

---

## Features

- **Custom Dataset**: Loads and parses Tic-Tac-Toe board data from a text file (`tic_tac_toe_data.txt`).
- **Neural Network Model**: A fully connected feedforward neural network to predict the best move from a given board state.
- **Training Loop**: The training loop uses the Adam optimizer and CrossEntropyLoss for classification, with dynamic learning rate scheduling and validation accuracy tracking.
- **Data Loaders**: Uses PyTorch’s `DataLoader` for batching and splitting the data into training and validation sets.

---

## Requirements

- Python 3.x
- PyTorch (torch)
- torch.utils.data (for data loading)
  
You can install the required packages using:

```bash
pip install torch
```

---

## Code Breakdown

### 1. **`TicTacToeDataset` Class**
   - **Purpose**: Loads the Tic-Tac-Toe game data from a text file and prepares it for training.
   - **Attributes**:
     - `__init__`: Reads the dataset from `tic_tac_toe_data.txt` and processes it into tuples of board state and optimal move.
     - `__len__`: Returns the number of samples in the dataset.
     - `__getitem__`: Returns a single data sample, consisting of the board (as a tensor) and the corresponding optimal move (as a tensor).

### 2. **Data Loading**

   - **Train and Validation Splitting**: The dataset is split into 80% training and 20% validation using `random_split`.
   - **DataLoader**: The `DataLoader` is used to batch the data during training, with shuffling for the training set and non-shuffling for the validation set.

   ```python
   train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
   val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
   ```

### 3. **Neural Network Model (`TicTacToeNN`)**

   - **Architecture**: A fully connected neural network with two hidden layers.
     - Input layer: 9 neurons (for the 9 board positions).
     - Two hidden layers: 16 neurons each.
     - Output layer: 9 neurons (one for each possible move).
   - **Activation**: ReLU for the hidden layers and Softmax for the output layer.
   - **Forward Method**: Defines how the data flows through the layers.

### 4. **Training and Evaluation**

   - **Training Loop**: The model is trained for a specified number of epochs. The optimizer and loss function are used to update the model’s weights.
     - The `CrossEntropyLoss` is used since this is a classification task.
     - The `Adam` optimizer is employed, with a learning rate scheduler to reduce the learning rate every 5 epochs.
   - **Validation**: After each epoch, the model's accuracy on the validation set is computed and printed. If the accuracy exceeds 90%, training stops early.

   ```python
   model.eval()  # Set model to evaluation mode
   ```

### 5. **Training Loop**

   - The training loop includes:
     - Backpropagation and weight updates for each batch.
     - Validation accuracy computation after each epoch.
     - Dynamic learning rate adjustment via the `StepLR` scheduler.
     - Early stopping if validation accuracy reaches or exceeds 90%.

---

## Usage

1. **Generate Data**:
   - Before training, you must generate the Tic-Tac-Toe dataset using the earlier data generation script. Ensure that the `tic_tac_toe_data.txt` file is available in the working directory.

2. **Train the Model**:
   - Run the loadData.py then the NNTicTacToe.py file to train the model.
   

3. **Monitoring**:
   - The training script will output the loss and validation accuracy after each epoch. If the validation accuracy reaches 90% or higher, the training will stop early.

---

## Example Output

```bash
Epoch 1, Loss: 2.3876, Validation Accuracy: 56.12%
Epoch 2, Loss: 1.6583, Validation Accuracy: 72.89%
Epoch 3, Loss: 1.2414, Validation Accuracy: 83.45%
Epoch 4, Loss: 0.9432, Validation Accuracy: 87.72%
Epoch 5, Loss: 0.7441, Validation Accuracy: 90.50%
Target reached!
```

---

## Notes
  
- **Evaluation**: After training, you can use the model to predict moves on new Tic-Tac-Toe boards.

---
