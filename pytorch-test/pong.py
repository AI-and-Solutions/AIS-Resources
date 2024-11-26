import gymnasium as gym
import torch
import torch.nn as nn
import torch.nn.functional as fn
import cv2 #OpenCV library, used to manipulate and process images
import numpy as np

environment = gym.make("ALE/Pong-v5")
environment.reset()

#Simplifying the data/observations for the agent
def preprocess_frame(frame):
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY) #Converts to grayscale to reduce each frame's data size - color takes up more space
    frame = cv2.resize(frame, (84, 84)) #Resize to 84x84
    return frame

state = preprocess_frame(state) #Transforms raw game frame to simplied data for the agent to read

#Creating a CNN to approximate Q-values; model will take in 4 frame stack and output set of Q-values- one for each action (4,84,84)
class DQN(nn.Module):
    def __init__(self, input_shape, num_actions):
        super(DQN, self).__init__()
        #Convolutional Layers - helps track spatial and temporal patterns - applies filters to detect low & high level features
        self.conv1 = nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4) #input shape (4,84,84); 32 filters; 8x8 grid of weights; 4 pixels at a time
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2) #Receives 32 channels as input, 64 ouput channels; detects unique feature each; filter 4x4, 2 pixels at a time
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1) 
        #Output flattened to 1d vector, passed through fully connected layers; helps network understand relationship between features and actions -> higher rewards
        self.fc1 = nn.Linear(7 * 7 * 64, 512) # First dense layer, first part is the expected vector size, second is number of nuerons - 512 learned features
        self.fc2 = nn.Linear(512, num_actions) #output layer, one Q-value per action
    #Defines the path data takes through the network
    def forward(self,x):
        x = fn.relu(self.conv1(x)) #Uses ReLU activation, sets any negative values to zero
        x = fn.relu(self.conv2(x)) #Passes output of conv1 through the 2nd convolutional layer and applies ReLU.
        x = fn.relu(self.conv3(x))
        x = x.view(x.size(0), -1) #Flatten for fully connected layers, flattens the data to shape of (batch_size,3136)
        x = fn.relu(self.fc1(x)) #Passes flattened input through fc1, with ReLU for non-linearity. Transforms 3136 features to 512 learned features
        return self.fc2(x) #Final Q-values for each action
    

# Test environment and preprocess_frame function
for _ in range(5):
    state = environment.reset()  # Reset the environment
    state = preprocess_frame(state)  # Preprocess the frame
    print(f"Processed frame shape: {state.shape}")  # Should output (84, 84)

