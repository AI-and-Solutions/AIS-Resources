import numpy as np
import pygame
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2  # For frame preprocessing

# Pong Game Environment
class PongGame:
    def __init__(self, screen_width=400, screen_height=300):
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.paddle_width = 10
        self.paddle_height = 50
        self.paddle_speed = 4
        self.ball_radius = 5
        self.reset()

    def reset(self):
        self.ball_pos = np.array([self.screen_width // 2, self.screen_height // 2])
        self.ball_vel = np.random.choice([-1, 1], size=2) * np.array([3, 2])  # Random velocity
        self.paddle1_pos = self.screen_height // 2 - self.paddle_height // 2
        self.paddle2_pos = self.screen_height // 2 - self.paddle_height // 2
        self.score = [0, 0]
        return self.get_frame()

    def step(self, action1, action2):
        # Update paddles
        self.paddle1_pos = np.clip(self.paddle1_pos + action1 * self.paddle_speed, 0, self.screen_height - self.paddle_height)
        self.paddle2_pos = np.clip(self.paddle2_pos + action2 * self.paddle_speed, 0, self.screen_height - self.paddle_height)

        # Update ball position
        self.ball_pos += self.ball_vel

        # Ball collision with walls
        if self.ball_pos[1] <= 0 or self.ball_pos[1] >= self.screen_height:
            self.ball_vel[1] *= -1

        # Ball collision with paddles
        if (self.ball_pos[0] <= self.paddle_width and
            self.paddle1_pos <= self.ball_pos[1] <= self.paddle1_pos + self.paddle_height) or \
           (self.ball_pos[0] >= self.screen_width - self.paddle_width and
            self.paddle2_pos <= self.ball_pos[1] <= self.paddle2_pos + self.paddle_height):
            self.ball_vel[0] *= -1

        # Scoring
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

        return self.get_frame(), reward, done

    def get_frame(self):
        # Create a simple frame for CNN processing
        frame = np.zeros((self.screen_height, self.screen_width), dtype=np.uint8)
        frame = cv2.rectangle(frame, (0, self.paddle1_pos), 
                              (self.paddle_width, self.paddle1_pos + self.paddle_height), 255, -1)
        frame = cv2.rectangle(frame, (self.screen_width - self.paddle_width, self.paddle2_pos), 
                              (self.screen_width, self.paddle2_pos + self.paddle_height), 255, -1)
        frame = cv2.circle(frame, tuple(self.ball_pos), self.ball_radius, 255, -1)
        return frame

    def render(self, screen):
        screen.fill((0, 0, 0))
        pygame.draw.rect(screen, (255, 255, 255), (0, self.paddle1_pos, self.paddle_width, self.paddle_height))
        pygame.draw.rect(screen, (255, 255, 255), 
                         (self.screen_width - self.paddle_width, self.paddle2_pos, self.paddle_width, self.paddle_height))
        pygame.draw.circle(screen, (255, 255, 255), self.ball_pos.astype(int), self.ball_radius)
        pygame.display.flip()

# Preprocessing for CNN input
def preprocess_frame(frame):
    frame = cv2.resize(frame, (84, 84))
    frame = np.expand_dims(frame, axis=0)  # Add channel dimension
    return frame.astype(np.float32) / 255.0  # Normalize

# DQN Model
class DQN(nn.Module):
    def __init__(self, input_shape, num_actions):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.fc1 = nn.Linear(7 * 7 * 64, 512)
        self.fc2 = nn.Linear(512, num_actions)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)

# Example usage
if __name__ == "__main__":
    pygame.init()
    screen = pygame.display.set_mode((400, 300))
    game = PongGame()
    model = DQN((1, 84, 84), num_actions=3)

    for episode in range(10):  # Example loop
        state = game.reset()
        done = False
        while not done:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    exit()
            action = np.random.choice([-1, 0, 1])  # Random action for now
            next_frame, reward, done = game.step(action, 0)
            preprocessed_state = preprocess_frame(next_frame)
            game.render(screen)