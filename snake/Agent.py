"""
Agent. Implement Deep Q network to learn playing snake.
"""

import torch
from .Snake import Snake
import numpy as np
import random
from matplotlib import pyplot as plt

NN_L1 = 24
NN_L2 = 150
NN_L3 = 100
NN_L4 = 4
LEARNING_RATE = 1e-4
GAMMA = 0.9

ACTION_SET = {
    0: "UP",
    1: "RIGHT",
    2: "LEFT",
    3: "DOWN",
}

EPSILON_INIT = 1.0
EPSILON_MIN = 0.05

class Agent():
    # Say that 300 step is an epoch
    # 300 step is sufficient to reach a 10-case long snake.
    STEP_PER_EPOCHS = 300

    def __init__(self):
        self.model = torch.nn.Sequential(
            torch.nn.Linear(NN_L1, NN_L2),
            torch.nn.ReLU(),
            torch.nn.Linear(NN_L2, NN_L3),
            torch.nn.ReLU(),
            torch.nn.Linear(NN_L3, NN_L4)
        )
        self.loss_fn = torch.nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=LEARNING_RATE)
        self.losses = []
        self.epsilon = EPSILON_INIT
        self.internal_step_counter = 0
        self.epoch_cumuled_reward = 0
        self.cumuled_rewards = []
        self.epsilons = []
        self.epoch_counter = 0

    def run_best_step(self, snake:Snake):
        state_ = np.array(snake.get_state()).astype(float) + np.random.rand(1,NN_L1)/100.0
        state = torch.from_numpy(state_).float()
        qval = self.model(state) #H
        qval_ = qval.data.numpy()
        action_ = np.argmax(qval_)
        action = ACTION_SET[int(action_)] #J
        snake.set_direction(action)
        snake.move()

    def run_step(self, snake:Snake):
        self.internal_step_counter += 1
        state_ = np.array(snake.get_state()).astype(float) + np.random.rand(1,NN_L1)/100.0
        state = torch.from_numpy(state_).float()
        qval = self.model(state) #H
        qval_ = qval.data.numpy()
        if (random.random() < self.epsilon): #I
            action_ = np.random.randint(0,4)
        else:
            action_ = np.argmax(qval_)
        action = ACTION_SET[int(action_)] #J
        snake.set_direction(action)
        snake.move()
        reward = snake.get_reward()
        self.epoch_cumuled_reward += reward
        # Y = torch.Tensor([reward]).detach()

        state2_ = np.array(snake.get_state()).astype(float) + np.random.rand(1,NN_L1)/100.0
        state2 = torch.from_numpy(state2_).float()
        with torch.no_grad():
            newQ = self.model(state2)
            #newQ = model(state2.reshape(1,64))
        maxQ = torch.max(newQ) #M
        Y = torch.tensor(reward + (GAMMA * maxQ), dtype=torch.float32)
        X = qval.squeeze()[action_]
        loss = self.loss_fn(X, Y)
        self.optimizer.zero_grad()
        loss.backward()
        self.losses.append(loss.item())
        self.optimizer.step()
        if self.internal_step_counter > self.STEP_PER_EPOCHS:
            # Terminate an epoch and start another
            self.cumuled_rewards += [self.epoch_cumuled_reward]
            self.epoch_cumuled_reward = 0
            self.internal_step_counter = 0
            # self.epsilon = max(EPSILON_MIN, self.epsilon - 0.002)
            self.epsilon = max(EPSILON_MIN, self.epsilon * 0.99)
            self.epsilons += [self.epsilon]
            self.epoch_counter += 1
            snake.reset()

    def plot_losses(self):
        plt.figure(figsize=(10,7))
        plt.xlabel("Steps", fontsize=22)
        plt.ylabel("Loss", fontsize=22)
        plt.plot(self.losses)

    def plot_rewards(self):
        plt.figure(figsize=(10,7))
        plt.xlabel(f"Epoch ({self.STEP_PER_EPOCHS} steps)", fontsize=22)
        plt.ylabel("Reward", fontsize=22)
        plt.plot(self.cumuled_rewards)

    def plot_epsilons(self):
        plt.figure(figsize=(10,7))
        plt.xlabel(f"Epoch", fontsize=22)
        plt.ylabel("Epsilon", fontsize=22)
        plt.plot(self.epsilons)

    def show(self):
        plt.show()

    def export_weight(self, filepath:str):
        torch.save(self.model.state_dict(), filepath)

    def import_weight(self, filepath:str):
        """
        Import weigth from a file.
        Model must match the nn architecture.
        """
        self.model.load_state_dict(torch.load(filepath))



