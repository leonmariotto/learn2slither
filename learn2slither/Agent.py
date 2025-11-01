"""
Agent. Implement Deep Q network to learn playing snake.
"""

import torch
from .Snake import Snake
from .MetaParameters import MetaParameters
import numpy as np
import random
from matplotlib import pyplot as plt
import collections

NN_L1 = 24
NN_L2 = 150
NN_L3 = 100
NN_L4 = 4

ACTION_SET = {
    0: "UP",
    1: "RIGHT",
    2: "LEFT",
    3: "DOWN",
}
VALIDATION_EPSILON = 0.0005
VALIDATION_RESET_FREQUENCY = 200


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
        - done: boolean, used to know if future state have to be taken into account. (if
            snake is dead there is no need to take new state into account)
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


class Agent:
    """
    A RL agent that play snake.
    Use a 2 layers sequential neural network with rectified linear unit as activation function.
    Use Mean-squared error as activation function.
    Use Adam optimizer.
    Use replay buffer.
    Use a target network different from main network, with a defined synchronization rate.
    Meta parameters comes from an external class at init.

    About torch :
    - torch.nn.Sequential is a container that chain multiple neural network and functions.
    - torch.nn.ReLU is: Rectified Linear Unit function between each layer: f(x) = max(0,x).
        thout this stacking Linear layer would collapse into a single linear transformation
        (no learning power). linear + linear = linear, nn.ReLU break the linearity.
    - torch.nn.Linear is: Affine linear function: y = x @ W.T + b, where x is the input vector,
        W.T the weight matrix, and b a bias vector (@ is a matrix multiplication).
    """

    def __init__(self, meta_parameters: MetaParameters):
        self.meta_parameters = meta_parameters
        # The main model is updated at each step.
        # The state is inputed to the model, a prediction of reward-action is outputed (Q-value)
        # so we can choose the highest reward-action of the Q-value.
        self.model = torch.nn.Sequential(
            torch.nn.Linear(NN_L1, NN_L2),
            torch.nn.ReLU(),
            torch.nn.Linear(NN_L2, NN_L3),
            torch.nn.ReLU(),
            torch.nn.Linear(NN_L3, NN_L4),
        )
        # This model is updated at a defined synchonization rate (meta-parameter)
        # where it is synchronized to main network.
        # So this network is more stable than main network.
        # This is needed to reduce learning instability, although the synchronization
        # rate should be chosen carefully.
        # This network is only used to compute future reward (γ * max(Q_target(S₄))) in Bellman equation.
        self.target_model = torch.nn.Sequential(
            torch.nn.Linear(NN_L1, NN_L2),
            torch.nn.ReLU(),
            torch.nn.Linear(NN_L2, NN_L3),
            torch.nn.ReLU(),
            torch.nn.Linear(NN_L3, NN_L4),
        )
        self.target_model.load_state_dict(self.model.state_dict())
        # Mean squared error.
        self.loss_fn = torch.nn.MSELoss()
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.meta_parameters.learning_rate
        )
        self.replay_buffer = ReplayBuffer(self.meta_parameters.replay_buffer_size)
        self.losses = []
        # Epsilon is used to slowly become deterministic.
        self.epsilon = self.meta_parameters.epsilon_init
        self.internal_step_counter = 0
        self.epoch_cumuled_reward = 0
        self.cumuled_rewards = []
        self.epsilons = []
        self.snake_sizes = []
        self.epoch_counter = 0

    def run_best_step(self, snake: Snake):
        """
        Do not learn, only play the best move predicted.
        """
        self.internal_step_counter += 1
        state_ = (
            np.array(snake.get_state()).astype(float) + np.random.rand(1, NN_L1) / 100.0
        )
        state = torch.from_numpy(state_).float()
        qval = self.model(state)  # H
        qval_ = qval.data.numpy()
        if random.random() < VALIDATION_EPSILON:  # I
            action_ = np.random.randint(0, 4)
        else:
            action_ = np.argmax(qval_)
        action = ACTION_SET[int(action_)]  # J
        snake.set_direction(action)
        snake.move()
        reward = snake.get_reward()
        self.epoch_cumuled_reward += reward
        if self.internal_step_counter > self.meta_parameters.step_per_epoch:
            self.snake_sizes += [len(snake.snake)]
            # Terminate an epoch and start another
            self.cumuled_rewards += [self.epoch_cumuled_reward]
            self.epoch_cumuled_reward = 0
            self.internal_step_counter = 0
            self.epoch_counter += 1
            if self.epoch_counter % VALIDATION_RESET_FREQUENCY:
                snake.reset()

    def run_step(self, snake: Snake):
        """
        Run a step and learn from it.
        Use replay buffer and target network.
        """
        self.internal_step_counter += 1
        state_ = (
            np.array(snake.get_state()).astype(float) + np.random.rand(1, NN_L1) / 100.0
        )
        state = torch.from_numpy(state_).float()
        qval = self.model(state)  # H
        qval_ = qval.data.numpy()
        if random.random() < self.epsilon:  # I
            action_ = np.random.randint(0, 4)
        else:
            action_ = np.argmax(qval_)
        action = ACTION_SET[int(action_)]  # J
        snake.set_direction(action)
        snake.move()
        reward = snake.get_reward()
        self.epoch_cumuled_reward += reward
        state2_ = (
            np.array(snake.get_state()).astype(float) + np.random.rand(1, NN_L1) / 100.0
        )
        state2 = torch.from_numpy(state2_).float()
        # Because we take into account next states, the "done" state is special, as there is no
        # next state to take into account.
        # The "done_batch" value is used to transform to 0 the "future rewards" part.
        done = reward == Snake.DEATH_REWARD
        self.replay_buffer.add(state, action_, reward, state2, done)

        if len(self.replay_buffer) > self.meta_parameters.replay_buffer_batch_size:
            minibatch = self.replay_buffer.sample(
                self.meta_parameters.replay_buffer_batch_size
            )
            state_batch, action_batch, reward_batch, next_state_batch, done_batch = zip(
                *minibatch
            )

            state_batch = torch.cat(state_batch)
            action_batch = torch.Tensor(action_batch)
            reward_batch = torch.Tensor(reward_batch)
            next_state_batch = torch.cat(next_state_batch)
            done_batch = torch.Tensor(done_batch)

            Q_values = self.model(state_batch)
            with torch.no_grad():
                # Use target model to predict future Q values.
                next_Q_values = self.target_model(next_state_batch)
            Q_target = reward_batch + self.meta_parameters.gamma * torch.max(
                next_Q_values, dim=1
            ).values * (1 - done_batch)

            loss = self.loss_fn(
                Q_values.gather(1, action_batch.long().unsqueeze(1)).squeeze(), Q_target
            )
            self.optimizer.zero_grad()
            # Perform backpropagation on every gradient of every neuron in every layer of neural network.
            loss.backward()
            self.losses.append(loss.item())
            # Once gradient are computed optimizer update weight, with respect to the learning weigth.
            self.optimizer.step()

        if self.internal_step_counter > self.meta_parameters.step_per_epoch:
            self.snake_sizes += [len(snake.snake)]
            # Terminate an epoch and start another
            self.cumuled_rewards += [self.epoch_cumuled_reward]
            self.epoch_cumuled_reward = 0
            self.internal_step_counter = 0
            # self.epsilon = max(EPSILON_MIN, self.epsilon - 0.002)
            if self.epoch_counter % self.meta_parameters.epsilon_update_freq == 0:
                self.epsilon = max(
                    self.meta_parameters.epsilon_min,
                    self.epsilon * self.meta_parameters.epsilon_a
                    + self.meta_parameters.epsilon_b,
                )
            self.epsilons += [self.epsilon]
            self.epoch_counter += 1
            if self.epoch_counter % self.meta_parameters.target_model_update_freq == 0:
                self.target_model.load_state_dict(self.model.state_dict())

    def plot_training_metrics(self):
        """
        Prepare training metrics.
        """
        fig, axs = plt.subplots(4, 1, figsize=(10, 18))
        axs[0].set_title("Losses", fontsize=22)
        axs[0].set_xlabel("Steps", fontsize=16)
        axs[0].set_ylabel("Loss", fontsize=16)
        axs[0].plot(self.losses)
        axs[1].set_title("Rewards", fontsize=22)
        axs[1].set_xlabel("Epoch", fontsize=16)
        axs[1].set_ylabel("Reward", fontsize=16)
        axs[1].plot(self.cumuled_rewards)
        axs[2].set_title("Epsilon", fontsize=22)
        axs[2].set_xlabel("Epoch", fontsize=16)
        axs[2].set_ylabel("Epsilon", fontsize=16)
        axs[2].plot(self.epsilons)
        axs[3].set_title("Snake Size", fontsize=22)
        axs[3].set_xlabel("Epoch", fontsize=16)
        axs[3].set_ylabel("Size", fontsize=16)
        axs[3].plot(self.snake_sizes)
        plt.tight_layout(pad=5.0)

    def plot_validation_metrics(self):
        """
        Prepare validation metrics.
        """
        fig, axs = plt.subplots(2, 1, figsize=(10, 18))
        axs[0].set_title("Rewards", fontsize=22)
        axs[0].set_xlabel("Epoch", fontsize=16)
        axs[0].set_ylabel("Reward", fontsize=16)
        axs[0].plot(self.cumuled_rewards)
        axs[1].set_title("Snake Size", fontsize=22)
        axs[1].set_xlabel("Epoch", fontsize=16)
        axs[1].set_ylabel("Size", fontsize=16)
        axs[1].plot(self.snake_sizes)
        plt.tight_layout(pad=5.0)

    def save_plots(self, filepath):
        """
        Save model metrics into filepath.
        Should be called after plot_metrics.
        """
        plt.savefig(filepath)

    def show(self):
        """
        Show model metrics.
        Should be called after plot_metrics.
        """
        plt.show()

    def export_weight(self, filepath: str):
        """
        Export model weight in filepath.
        """
        torch.save(self.model.state_dict(), filepath)

    def import_weight(self, filepath: str):
        """
        Import weigth from a file.
        Model must match the nn architecture.
        """
        self.model.load_state_dict(torch.load(filepath))
        self.target_model.load_state_dict(torch.load(filepath))
