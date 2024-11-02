import gym
import torch
import torch.nn as nn
import torch.nn.functional as fn
import cv2
import numpy as np

environment = gym.make("PongNoFrameskip-v4")
state = environment.reset()  #Resets back to starting conditions after each episode

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
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        #Output flattened to 1d vector, passed through fully connected layers; helps network understand relationship between features and actions -> higher rewards
        self.fc1 = nn.Linear(7 * 7 * 64, 512) # First dense layer
        self.fc2 = nn.Linear(512, num_actions) #output layer, one Q-value per action



