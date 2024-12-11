import numpy as np
import pygame
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2  # For frame preprocessing
import random
from collections import deque #Double-ended queue; used to implement the replay buffer for storing past experiences efficiently.

# Pong Game Environment
class PongGame:
    #Initializes game parameters and game state
    def __init__(self, screen_width=400, screen_height=300):
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.paddle_width = 10
        self.paddle_height = 50
        self.paddle_speed = 4
        self.ball_radius = 5
        self.reset()

    #Resets game state
    def reset(self):
        self.ball_pos = np.array([self.screen_width // 2, self.screen_height // 2]) #Ball starts at center of screen
        self.ball_vel = np.random.choice([-1, 1], size=2) * np.array([3, 2])  # Random velocity
        self.paddle1_pos = self.screen_height // 2 - self.paddle_height // 2
        self.paddle2_pos = self.screen_height // 2 - self.paddle_height // 2
        self.score = [0, 0]
        return self.get_frame() #Returns the initial game frame for display or processing

    #Control the vertical movement of paddles (-1 = up, 1 = down, 0 = stationary)
    def step(self, action1, action2):
        # Update paddles
        self.paddle1_pos = np.clip(self.paddle1_pos + action1 * self.paddle_speed, 0, self.screen_height - self.paddle_height)
        self.paddle2_pos = np.clip(self.paddle2_pos + action2 * self.paddle_speed, 0, self.screen_height - self.paddle_height)

        #Update ball position
        self.ball_pos += self.ball_vel

        #Ball collision with walls
        if self.ball_pos[1] <= 0 or self.ball_pos[1] >= self.screen_height:
            self.ball_vel[1] *= -1

        #Ball collision with paddles
        if (self.ball_pos[0] <= self.paddle_width and
            self.paddle1_pos <= self.ball_pos[1] <= self.paddle1_pos + self.paddle_height) or \
           (self.ball_pos[0] >= self.screen_width - self.paddle_width and
            self.paddle2_pos <= self.ball_pos[1] <= self.paddle2_pos + self.paddle_height):
            self.ball_vel[0] *= -1

        #Scoring
        reward = 0
        done = False
        if self.ball_pos[0] < 0:  # Player 2 scores
            self.score[1] += 1
            reward = -1
            self.reset()
        elif self.ball_pos[0] > self.screen_width:  # Player 1 scores
            self.score[0] += 1
            reward = 1
            self.reset() 

        return self.get_frame(), reward, done #Returns current game frame and reward for last action (1 or -1)
    
    #Create a simple grayscale frame for CNN processing
    def get_frame(self):
        frame = np.zeros((self.screen_height, self.screen_width), dtype=np.uint8)
        frame = cv2.rectangle(frame, (0, self.paddle1_pos), 
                              (self.paddle_width, self.paddle1_pos + self.paddle_height), 255, -1)
        frame = cv2.rectangle(frame, (self.screen_width - self.paddle_width, self.paddle2_pos), 
                              (self.screen_width, self.paddle2_pos + self.paddle_height), 255, -1)
        frame = cv2.circle(frame, tuple(self.ball_pos), self.ball_radius, 255, -1)
        return frame

    #Draws paddles and ball on screen for real-time viewing
    def render(self, screen):
        screen.fill((0, 0, 0))
        pygame.draw.rect(screen, (255, 255, 255), (0, self.paddle1_pos, self.paddle_width, self.paddle_height))
        pygame.draw.rect(screen, (255, 255, 255), 
                         (self.screen_width - self.paddle_width, self.paddle2_pos, self.paddle_width, self.paddle_height))
        pygame.draw.circle(screen, (255, 255, 255), self.ball_pos.astype(int), self.ball_radius)
        pygame.display.flip() #Updates the display

#Preprocessing for CNN input
def preprocess_frame(frame):
    frame = cv2.resize(frame, (84, 84)) #Resizes frames to a standard 84x84 resolution to reduce input size
    frame = np.expand_dims(frame, axis=0)  #Add channel dimension
    return frame.astype(np.float32) / 255.0  #Normalize pixel values from [0, 255] to [0,1]

#DQN Model
class DQN(nn.Module):
    def __init__(self, input_shape, num_actions):
        super(DQN, self).__init__()
        #Convolutional Layers: Extract spatial features from frames
        self.conv1 = nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        #Fully Connected Layers: Process extracted features to predict Q-values for all actions
        self.fc1 = nn.Linear(7 * 7 * 64, 512)
        self.fc2 = nn.Linear(512, num_actions)
    
    #Forward pass through the network
    def forward(self, x):
        #Applies ReLU activations after each convolutional and fully connected layer
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return self.fc2(x) #Predicts Q-values for all possible actions in the final layer.


#Hyperparameters
BATCH_SIZE = 32 #Determines the number of experiences sampled from the replay buffer in each training step

GAMMA = 0.99   #Discount factor; Controls the importance of future rewards in the Q-value calculation

EPSILON_START = 1.0 #Initial exploration rate in the epsilon-greedy strategy

EPSILON_END = 0.1 #Minimum exploration rate

EPSILON_DECAY = 10000 #Number of steps for epsilon to decay 

LEARNING_RATE = 1e-4 #Controls how much the neural network's weights are adjusted during training

TARGET_UPDATE_FREQUENCY = 1000 #Update target network every 1000 steps;specifies how often the target network is updated with the weights of the main Q-network

MEMORY_SIZE = 50000 #Maximum number of experiences stored in the replay buffer

NUM_EPISODES = 1000 #Number of episodes to train the agent



#Replay Buffer: Stores past experiences to sample for training
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity) #deque with a fixed max sixe automatically discards the oldest experiences when full

    #Adds a new experience to the replay buffer 
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    #Samples a batch of experiences
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size) #Randomly selects batch_size experiences
        states, actions, rewards, next_states, dones = zip(*batch)
        return (np.array(states), np.array(actions), np.array(rewards), #Returns states, actions, rewards, next states, and done flags as separate arrays
                np.array(next_states), np.array(dones))

    #Returns current size of buffer
    def __len__(self):
        return len(self.buffer)
    
# Initialize models, optimizer, and replay buffer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") #Determines whether to use a GPU or CPU for computations
main_model = DQN((1, 84, 84), num_actions=3).to(device) #Creates the main DQN model and moves it to the chosen device
target_model = DQN((1, 84, 84), num_actions=3).to(device) #Creates the target network with the same structures as the main model and moves it to the device
target_model.load_state_dict(main_model.state_dict()) #Initializes the target model with the same weights as the main model
target_model.eval() #Sets the target model to evaluation mode

optimizer = torch.optim.Adam(main_model.parameters(), lr=LEARNING_RATE) #Sets up the optimizer for updating the weights of the main model
replay_buffer = ReplayBuffer(MEMORY_SIZE) #Creates a replay buffer to store experience tuples during training

#Epsilon-greedy strategy
def epsilon_greedy_policy(state, epsilon): #Defines a policy for choosing actions based on the epsilon-greedy strategy
    if random.random() < epsilon: #Implements the exploration part of epsilon-greedy
        return random.randint(0, 2)  #Random action
    
    else: # Converts the current state into a tensor for input into the main model
        state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0) #Converts state (a NumPy array) into a PyTorch tensor of type float32 and moves it to the specified device
        q_values = main_model(state) #Computes Q-values for all possible actions from the current state
        return q_values.argmax().item() #Selects the action with the highest Q-value (exploitation)

#Train function    
def train_step():
    if len(replay_buffer) < BATCH_SIZE:
        return
    #Sample a batch from replay buffer
    states, actions, rewards, next_states, dones = replay_buffer.sample(BATCH_SIZE)
    states = torch.tensor(states, dtype = torch.float32, device = device)
    actions = torch.tensor(actions, dtype=torch.long, device = device)
    rewards = torch.tensor(rewards, dtype=torch.float32, device = device)
    next_states = torch.tensor(next_states, dtype=torch.float32, device=device)
    dones = torch.tensor(dones, dtype=torch.float32, device=device)
    
    #Computing predicted Q-values
    q_values = main_model(states)
    q_values = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)
    
    #Computing target Q-values
    with torch.no_grad():
        next_q_values = target_model(next_states).max(1)[0]
        target_q_values = rewards + GAMMA * next_q_values * (1-dones)

    #Computing loss
    loss = F.mse_loss(q_values, target_q_values)

    #Optimizing the model
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# Training loop
epsilon = EPSILON_START
step_count = 0
        


##Testing Pong Game Functionality##

#Game Initialization
game = PongGame()
frame = game.reset()
print(f"Initial frame shape: {frame.shape}")

#Simple random loop
for _ in range(100):
    action1 = random.choice([-1,0,1])
    action2 = random.choice([-1,0,1])
    frame, reward, done = game.step(action1, action2)
    print(f"Frame shape: {frame.shape}, Reward: {reward}")

#Checking rendering
pygame.init()
screen = pygame.display.set_mode((400, 300))

for _ in range(100):
    game.step(random.choice([-1, 0, 1]), random.choice([-1, 0, 1]))
    game.render(screen)
    pygame.time.delay(30)
pygame.quit()

#Testing DQN
model = DQN((1, 84, 84), num_actions=3)
dummy_input = torch.rand((1, 1, 84, 84))
output = model(dummy_input)
print(f"Output shape: {output.shape}")

#Checking parameter update
loss_fn = nn.MSELoss()
output = model(dummy_input)
target = torch.rand_like(output)
loss = loss_fn(output, target)
loss.backward()
optimizer.step()
print("Parameter update complete.")