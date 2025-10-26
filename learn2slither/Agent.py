"""
Agent. Implement Deep Q network to learn playing snake.
"""

import torch
from .Snake import Snake
import numpy as np
import random
from matplotlib import pyplot as plt
import collections


NN_L1 = 24
NN_L2 = 150
NN_L3 = 100
NN_L4 = 4

# The gradient tells you which way is downhill.
# The learning rate tells you how big a step you take in that direction.
# If your steps are too big you stumble around, if too low you crawl forever.
LEARNING_RATE = 1e-4

# Gamma meta parameter is applied in Bellman equation, it's the future reward discount factor.
# It determines how much future rewards matter compared to the current one.
# The model have the ability to think with many-step plan in mind, as (γ * max(Q_target(S₄))) can propagate
# to a 5 step away reward. But in this case the 5 step away reward will appear discounted by
# GAMMA ^ 5 => 0.9 ^ 5 => 0.59
# With a value of 0 only current reward matter, a value of 0.99 focus strongly on future rewards.
GAMMA = 0.9

# With a large replay buffer, rare transition (eating a green apple) are more used in training.
# This add more stability to training.
REPLAY_BUFFER_SIZE = 6000

# Control how many experiences are sampled from the replay buffer for each update. A large value
# means smoother gradient but slow updates (so slow the training process).
BATCH_SIZE = 600

TARGET_UPDATE_FREQUENCY = 3 # in epochs

ACTION_SET = {
    0: "UP",
    1: "RIGHT",
    2: "LEFT",
    3: "DOWN",
}

EPSILON_INIT = 1.0
EPSILON_MIN = 0.1

class ReplayBuffer:
    """
    Used to store the replay buffer list.
    self.buffer is a ring buffer, if full the last entry is pop
    when a new entry is append.

    Replay duffer data entry contain:
        - state: current state before action
        - action: the action chosen by model or picked random (with respect to epsilon)
        - reward: obtained reward following the chosen action
        - next_state: new state after action.
        - done: boolean, used to know if future state have to be taken into account. (if snake is dead there is
            no need to take new state into account)
    """
    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity)

    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        """
        Get a batch_size sample of replay entry.
        """
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)


class Agent():
    """
    A RL agent that play snake.
    Use a 2 layers sequential neural network with rectified linear unit as activation function.
    Use Mean-squared error as activation function.
    Use Adam optimizer.
    Use replay buffer.
    Use a target network different from main network, with a defined synchronization rate.
    """
    # Say that 300 step is an epoch
    # 300 step is sufficient to reach a 10-case long snake.
    STEP_PER_EPOCHS = 300

    def __init__(self):
        # The main model is used to make decision.
        # The state is inputed to the model, a prediction of reward-action is outputed (Q-value)
        # so we can choose the highest reward-action of the Q-value.
        self.model = torch.nn.Sequential(
            torch.nn.Linear(NN_L1, NN_L2),
            torch.nn.ReLU(),
            torch.nn.Linear(NN_L2, NN_L3),
            torch.nn.ReLU(),
            torch.nn.Linear(NN_L3, NN_L4)
        )
        # This model is updated every step but not used to make decision.
        # At a defined synchonization rate (meta-parameter) the target network is
        # synchronized to main network.
        # This is needed to reduce learning instability, although the synchronization
        # rate should be chosen carefully.
        self.target_model = torch.nn.Sequential(
            torch.nn.Linear(NN_L1, NN_L2),
            torch.nn.ReLU(),
            torch.nn.Linear(NN_L2, NN_L3),
            torch.nn.ReLU(),
            torch.nn.Linear(NN_L3, NN_L4)
        )
        self.target_model.load_state_dict(self.model.state_dict())
        # Mean squared error.
        self.loss_fn = torch.nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=LEARNING_RATE)
        self.replay_buffer = ReplayBuffer(REPLAY_BUFFER_SIZE)
        self.losses = []
        # Epsilon is used to slowly become deterministic.
        # The EPSILON_INIT, EPSILON min and epsilon decreasing rate are meta-parameters.
        self.epsilon = EPSILON_INIT
        self.internal_step_counter = 0
        self.epoch_cumuled_reward = 0
        self.cumuled_rewards = []
        self.epsilons = []
        self.snake_sizes = []
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
        state2_ = np.array(snake.get_state()).astype(float) + np.random.rand(1, NN_L1) / 100.0
        state2 = torch.from_numpy(state2_).float()
        # Because we take into account next states, the "done" state is special, as there is no
        # next state to take into account.
        # The "done_batch" value is used to transform to 0 the GAMMA part.
        done = reward == Snake.DEATH_REWARD  # Assuming DEATH_REWARD is -100
        self.replay_buffer.add(state, action_, reward, state2, done)

        if len(self.replay_buffer) > BATCH_SIZE:
            minibatch = self.replay_buffer.sample(BATCH_SIZE)
            state_batch, action_batch, reward_batch, next_state_batch, done_batch = zip(*minibatch)

            state_batch = torch.cat(state_batch)
            action_batch = torch.Tensor(action_batch)
            reward_batch = torch.Tensor(reward_batch)
            next_state_batch = torch.cat(next_state_batch)
            done_batch = torch.Tensor(done_batch)

            Q_values = self.model(state_batch)
            with torch.no_grad():
                next_Q_values = self.target_model(next_state_batch)
            Q_target = reward_batch + GAMMA * torch.max(next_Q_values, dim=1).values * (1 - done_batch)

            loss = self.loss_fn(Q_values.gather(1, action_batch.long().unsqueeze(1)).squeeze(), Q_target)
            self.optimizer.zero_grad()
            # Perform backpropagation on every gradient of every neuron in every layer of neural network.
            loss.backward()
            self.losses.append(loss.item())
            # Once gradient are computed optimizer update weight, with respect to the learning weigth.
            self.optimizer.step()

        if self.internal_step_counter > self.STEP_PER_EPOCHS:
            self.snake_sizes += [len(snake.snake)]
            # Terminate an epoch and start another
            self.cumuled_rewards += [self.epoch_cumuled_reward]
            self.epoch_cumuled_reward = 0
            self.internal_step_counter = 0
            # self.epsilon = max(EPSILON_MIN, self.epsilon - 0.002)
            self.epsilon = max(EPSILON_MIN, self.epsilon * 0.99)
            self.epsilons += [self.epsilon]
            self.epoch_counter += 1
            if self.epoch_counter % TARGET_UPDATE_FREQUENCY == 0:
                self.target_model.load_state_dict(self.model.state_dict())
            snake.reset()

    def plot_metrics(self):
        fig, axs = plt.subplots(4, 1, figsize=(10, 7))

        axs[0].set_title("Losses", fontsize=22)
        axs[0].set_xlabel("Steps", fontsize=16)
        axs[0].set_ylabel("Loss", fontsize=16)
        axs[0].plot(self.losses)
        axs[1].set_title("Rewards", fontsize=22)
        axs[1].set_xlabel(f"Epoch ({self.STEP_PER_EPOCHS} steps)", fontsize=16)
        axs[1].set_ylabel("Reward", fontsize=16)
        axs[1].plot(self.cumuled_rewards)
        axs[2].set_title("Epsilon", fontsize=22)
        axs[2].set_xlabel(f"Epoch", fontsize=16)
        axs[2].set_ylabel("Epsilon", fontsize=16)
        axs[2].plot(self.epsilons)
        axs[3].set_title("Snake Size", fontsize=22)
        axs[3].set_xlabel(f"Epoch", fontsize=16)
        axs[3].set_ylabel("Size", fontsize=16)
        axs[3].plot(self.snake_sizes)

    def save_plots(self, filepath):
        plt.tight_layout()
        plt.savefig(filepath, dpi=300, bbox_inches="tight")

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
        self.target_model.load_state_dict(torch.load(filepath))



