from pathlib import Path #Tool for working with paths on computer; easier to create and manage folders and files
from matplotlib import pyplot #Plots and displays images
import numpy as np #Library for handling arrays and math operatoins
import requests #Dowloads files from the internet
import pickle #Library for serializing and loading Python objects from files
import gzip #Work with compressed files
import torch #Library for machine learning with PyTorch, helping with tensors
import math
from IPython.core.debugger import set_trace
import torch.nn.functional as F
from torch import nn

#Sets up directories for data
DATA_PATH = Path("data") #A Path object representing the "data" folder
PATH = DATA_PATH / "mnist" #Points to the "mnist" folder inside the "data" folder
#The "/" operator joins folder names, making "data/mnist"

#Creates the data folder
PATH.mkdir(parents = True, exist_ok = True) #Creates the "data/mnist" folder; if parent folder like "data" doesn't exist, it will create it too; avoids error if folder exists

#Setting up download link and file
URL = "https://github.com/pytorch/tutorials/raw/main/_static/" #data address where dataset (mnist) is stored
FILENAME = "mnist.pkl.gz" #name of compressed dataset file

#Dowloading the dataset if not already there
if not (PATH / FILENAME).exists():  #If file (mnist.pkl.gz) is not already downloaded
    content = requests.get(URL + FILENAME).content #Downloads dataset and holds raw data of the downloaded file from internet
    (PATH / FILENAME).open("wb").write(content) #opens file for writing in binary mode and saves downloaded content into file


#Loading the data from the compressed file
with gzip.open((PATH / FILENAME).as_posix(), "rb") as f: #opens the compressed file and converts the path to a string that works on any operating system
    ((x_train, y_train), (x_valid, y_valid), _) = pickle.load(f, encoding="latin-1") #loads the dataset from the compressed file
    #x_train - training images (as arrays)
    #y_train - training labels (numbers from 0-9 representing which digit the images shows)
    #x_valid/y_valid - Validation images and labels


#Displaying the first training image
pyplot.imshow(x_train[0].reshape((28,28)), cmap="gray") #x_train[0] - first image from training set; .reshape - reshapes flat array into a 28x28 grid; image is greyscaled

#Checks if code is running in Google Colab
try:
    import google.colab

except ImportError:
    pyplot.show()

#Prints shape of training image    
print(x_train.shape)

#Converts data to PyTorch Tensors
x_train, y_train, x_valid, y_valid = map(torch.sensor, (x_train, y_train, x_valid, y_valid)) #map(): Applies torch.sensor() to each of the data sets, converting them from Numpy arrays to Pytorch tensors

#Storing the shape of the training data
n, c = x_train.shape #returns the shape of the tensor (n = number of training examples(ex 50,000 images), c = number of features (ex 784 pixels per image))
print(x_train, y_train) #Prints data for inspection
print(x_train.shape) #Prints the shape of x_train
print(y_train.min(), y_train.max()) #prints the min and max label values (should be 0 and 9)

weights = torch.randn(784,10) / math.sqrt(784) #Initializes the weight tensors with random values from a normal distribution(mean=0,std=1); 784 number of features (28x28image); 10 number of output classes (0-9)
weights.requires_grad_() #Tells PyTorch to track gradients for this tensor using back propagation
bias = torch.zeros(10, requires_grad=True) #Initializes bias tensor (10 values, one for each class) with zeros, also tracks gradients

#Log softmax function - converts raw predicitons into probabilities and returns the log of it
def log_softmax(x):
    return x - x.exp(),sum(-1).log().unsqueeze(-1)

#Model Function - takes a batch of inputs (xb) and returns log-softmax predictions
def model(xb):
    return log_softmax(xb @ weights + bias) #Matrix multiplication between input xb and weights; adds the bias to the result before applying log

#Batch size and predictions
bs = 64 #Batch size

xb = x_train[0:bs] #Takes the first 64 training examples
preds = model(xb) #Get predictions for the batch
preds[0], preds.shape #Print the first prediction and shape of all predictions
print(preds[0], preds.shape)

#Negative log likelihood loss function - selects the predicted log probability for the correct class from each example in the batch and compute the mean loss
#Used to evaluate how well the model is predicting the correct classes
def nll(input, target):
    return -input[range(target.shape[0]), target].mean()

loss_func = nll

yb = y_train[0:bs]
print(loss_func(preds, yb))

def accuracy(out, yb):
    preds = torch.argmax(out, dim=1)
    return (preds == yb).float().mean()

print(accuracy(preds,yb))

lr = 0.5  # learning rate
epochs = 2  # how many epochs to train for

for epoch in range(epochs):
    for i in range((n - 1) // bs + 1):
        set_trace()
        start_i = i * bs
        end_i = start_i + bs
        xb = x_train[start_i:end_i]
        yb = y_train[start_i:end_i]
        pred = model(xb)
        loss = loss_func(pred, yb)

        loss.backward()
        with torch.no_grad():
            weights -= weights.grad * lr
            bias -= bias.grad * lr
            weights.grad.zero_()
            bias.grad.zero_()

print(loss_func(model(xb), yb), accuracy(model(xb), yb))

loss_func = F.cross_entropy

def model(xb):
    return xb @ weights + bias

print(loss_func(model(xb), yb), accuracy(model(xb), yb))

class Mnist_Logistic(nn.Module):
    def __init__(self):
        super().__init__()
        self.weights = nn.Parameter(torch.randn(784, 10) / math.sqrt(784))
        self.bias = nn.Parameter(torch.zeros(10))

    def forward(self, xb):
        return xb @ self.weights + self.bias
    
model = Mnist_Logistic()

print(loss_func(model(xb), yb))

with torch.no_grad():
    for p in model.parameters(): p -= p.grad * lr
    model.zero_grad()

def fit():
    for epoch in range(epochs):
        for i in range((n - 1) // bs + 1):
            start_i = i * bs
            end_i = start_i + bs
            xb = x_train[start_i:end_i]
            yb = y_train[start_i:end_i]
            pred = model(xb)
            loss = loss_func(pred, yb)

            loss.backward()
            with torch.no_grad():
                for p in model.parameters():
                    p -= p.grad * lr
                model.zero_grad()

fit()

print(loss_func(model(xb), yb))
