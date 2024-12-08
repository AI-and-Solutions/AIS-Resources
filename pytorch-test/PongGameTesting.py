import numpy as np #library for numerical computing, used here for array manipulation and random number generation.
import pygame #library for creating games and rendering graphics
import torch #Building a training deep learning models
import torch.nn as nn
import torch.nn.functional as F
import cv2 #Imports OpenCV, a library for image processing, used for resizing and normalizing frames.
import random #Generates random numbers 
from collections import deque #Fast queue like container, used for replay buffer
import matplotlib.pyplot as plt #plots training performance graphs
import csv #Logging collected data
import os

#Pong Game Environment / The Physics Engine
class PongGame:
    def __init__(self, screen_width=400, screen_height=300):
        pygame.init()
        #Dimensions for screen, paddles, and ball
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.paddle_width = 10
        self.paddle_height = 50
        self.paddle_speed = 4
        self.ball_radius = 5
        #Creates titled pygame window of specified dimensions
        self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
        pygame.display.set_caption("Pong Game")
        self.clock = pygame.time.Clock() #Initializes clock for managing the game frame rate
        self.reset()

    #Resets game state
    def reset(self):
        self.ball_pos = np.array([self.screen_width // 2, self.screen_height // 2]) #Ball starts at center of screen
        self.ball_vel = np.random.choice([-1, 1], size=2) * np.array([3, 2]) #Random initial direction for ball
        self.paddle1_pos = self.screen_height // 2 - self.paddle_height // 2 #both paddles at the vertical center of the screen.
        self.paddle2_pos = self.screen_height // 2 - self.paddle_height // 2
        self.score = [0, 0]
        return preprocess_frame(self.get_frame()) #Returns a preprocessed image of the current game frame
    
    #Updates paddles position, ball position, and checks for collisions
    def step(self, action1, action2):
        self.paddle1_pos = np.clip(
            self.paddle1_pos + action1 * self.paddle_speed, 0, self.screen_height - self.paddle_height
        )
        self.paddle2_pos = np.clip(
            self.paddle2_pos + action2 * self.paddle_speed, 0, self.screen_height - self.paddle_height
        )
        self.ball_pos += self.ball_vel

        if self.ball_pos[1] <= 0 or self.ball_pos[1] >= self.screen_height: #Reverses balls vertical direction
            self.ball_vel[1] *= -1

        if (self.ball_pos[0] <= self.paddle_width and #Reverses ball horizontal direction
            self.paddle1_pos <= self.ball_pos[1] <= self.paddle1_pos + self.paddle_height) or \
           (self.ball_pos[0] >= self.screen_width - self.paddle_width and
            self.paddle2_pos <= self.ball_pos[1] <= self.paddle2_pos + self.paddle_height):
            self.ball_vel[0] *= -1

        #Initializes reward to 0 and game termination flag
        reward = 0
        done = False
        if self.ball_pos[0] < 0:  #Player 2 scores
            reward = -1
            self.reset()
            done = True
        elif self.ball_pos[0] > self.screen_width:  #Player 1 scores
            reward = 1
            self.reset()
            done = True

        return preprocess_frame(self.get_frame()), reward, done #Returns the processed game frame, reward for the current step, and state of game


    #Game frame
    def get_frame(self):
        frame = np.zeros((self.screen_height, self.screen_width), dtype=np.uint8) #Background
        cv2.rectangle(frame, (0, self.paddle1_pos), #Paddles
                      (self.paddle_width, self.paddle1_pos + self.paddle_height), 255, -1)
        cv2.rectangle(frame, (self.screen_width - self.paddle_width, self.paddle2_pos),
                      (self.screen_width, self.paddle2_pos + self.paddle_height), 255, -1)
        cv2.circle(frame, tuple(self.ball_pos), self.ball_radius, 255, -1) #Ball
        return frame

    #Renders game on Pygame
    def render(self):
        self.screen.fill((0, 0, 0))
        pygame.draw.rect(self.screen, (255, 255, 255),
                         (0, self.paddle1_pos, self.paddle_width, self.paddle_height))
        pygame.draw.rect(self.screen, (255, 255, 255),
                         (self.screen_width - self.paddle_width, self.paddle2_pos, self.paddle_width, self.paddle_height))
        pygame.draw.circle(self.screen, (255, 255, 255), self.ball_pos.tolist(), self.ball_radius)
        pygame.display.flip()
        self.clock.tick(60)

#Preprocessing for CNN input
def preprocess_frame(frame):
    frame = cv2.resize(frame, (84, 84)) #Resizes game fram to 84x84
    frame = np.expand_dims(frame, axis=0)
    return frame.astype(np.float32) / 255.0 #Converts range from 0 - 255 to 0 - 1

#DQN Model
class DQN(nn.Module):
    def __init__(self, input_shape, num_actions):
        super(DQN, self).__init__()
        #Convolutional layers
        self.conv1 = nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        #Fully connected layers; takesw flattened output, outputs 512 features
        self.fc1 = nn.Linear(7 * 7 * 64, 512)
        self.fc2 = nn.Linear(512, num_actions)

    #Applies convolutional layers and Relu activation to the input
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x)) #Flattens output of CN layers into fully connected layers and returns Q values of each action
        return self.fc2(x)

#Replay Buffer; stores past experiences
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (np.array(states), np.array(actions), np.array(rewards),
                np.array(next_states), np.array(dones))

    def __len__(self):
        return len(self.buffer)


# Hyperparameters
BATCH_SIZE = 32 #Number of training samples
GAMMA = 0.99
EPSILON_START = 1.0
EPSILON_END = 0.1
EPSILON_DECAY = 10000
LEARNING_RATE = 1e-4
TARGET_UPDATE_FREQUENCY = 1000
MEMORY_SIZE = 50000
NUM_EPISODES = 10
LOG_FILE = "training_log.csv"

#Training setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
main_model = DQN((1, 84, 84), num_actions=3).to(device)
target_model = DQN((1, 84, 84), num_actions=3).to(device)
target_model.load_state_dict(main_model.state_dict())
target_model.eval()
optimizer = torch.optim.Adam(main_model.parameters(), lr=LEARNING_RATE)
replay_buffer = ReplayBuffer(MEMORY_SIZE)

#Logging setup
if not os.path.exists(LOG_FILE):
    with open(LOG_FILE, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Episode", "Total Reward", "Epsilon"])

#Visualization
plt.ion()
fig, ax = plt.subplots()
rewards_plot, = ax.plot([], label="Total Reward")
ax.legend()
ax.set_xlabel("Episode")
ax.set_ylabel("Reward")
plt.title("Training Progress")

#Training loop
game = PongGame()
epsilon = EPSILON_START
rewards_history = []

for episode in range(NUM_EPISODES):
    state = game.reset()
    total_reward = 0
    done = False

    while not done:
        game.render()
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                exit()

        action = random.randint(0, 2) if random.random() < epsilon else main_model(torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)).argmax().item()
        next_state, reward, done = game.step(action - 1, random.choice([-1, 0, 1]))
        replay_buffer.push(state, action, reward, next_state, done)
        state = next_state
        total_reward += reward

        if len(replay_buffer) >= BATCH_SIZE:
            states, actions, rewards, next_states, dones = replay_buffer.sample(BATCH_SIZE)
            states = torch.tensor(states, dtype=torch.float32, device=device)
            actions = torch.tensor(actions, dtype=torch.long, device=device)
            rewards = torch.tensor(rewards, dtype=torch.float32, device=device)
            next_states = torch.tensor(next_states, dtype=torch.float32, device=device)
            dones = torch.tensor(dones, dtype=torch.float32, device=device)

            q_values = main_model(states).gather(1, actions.unsqueeze(1)).squeeze(1)
            with torch.no_grad():
                next_q_values = target_model(next_states).max(1)[0]
                target_q_values = rewards + GAMMA * next_q_values * (1 - dones)

            loss = F.mse_loss(q_values, target_q_values)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    rewards_history.append(total_reward)
    epsilon = max(EPSILON_END, EPSILON_START - episode / EPSILON_DECAY)

    if episode % TARGET_UPDATE_FREQUENCY == 0:
        target_model.load_state_dict(main_model.state_dict())

    #Update visualization
    rewards_plot.set_data(range(len(rewards_history)), rewards_history)
    ax.relim()
    ax.autoscale_view()
    plt.pause(0.01)

    #Log episode
    with open(LOG_FILE, mode="a", newline="") as file:
        writer = csv.writer(file)
        writer.writerow([episode, total_reward, epsilon])

    print(f"Episode {episode}, Reward: {total_reward}, Epsilon: {epsilon:.3f}")

#Plot
plt.ioff()
plt.show()