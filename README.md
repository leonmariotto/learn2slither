# Learn2Slither

Reinforcement learning project implementing a Deep Q Network to beat
a snake game.

## Usage

Run `uv run agent.py --help` for a list of available option.
If you want to play snake just run `uv run play_snake.py`.

### Install

Use uv to pull necessary packages.
```
uv sync
```

### Run trained model

There is a mode to see to agent playing, option `--play`:
```
uv run agent.py -i models/v5.weight --play
```
There is a mode to evaluate an agent, option `--validate`,
it produce graph that we can use to see the efficiency :
```
uv run agent.py -i models/v5.weight --validate -d
```

### Training

A model can be loaded using `-i` option, the trained model can be output
using `-o` option.
To re-train a model one can use :
```
uv run agent.py -i v1.weight -o v1.weight -t 600
```
The `-t` option specify the time of training (in second, so 10mn in this case).
The `-mp` option is used to set meta-parameters through yaml file.
```
uv run agent.py -i v5.weight -o v5.weight -mp meta/v5.yml -t 600
```

## How it work ?

This is a Deep Q Network. 
The model take the state of Snake in input and output a Q value prediction of action-rewards.

### Model Shape

We have this :
```
Linear(24, 150) → ReLU → Linear(150, 100) → ReLU → Linear(100, 4)
```
24 (input) and 4 (output) are fixed.
Hidden layer size control model capacity, large model can learn more but can be difficult to train.
Typical architecture start with a wider first layer (150) for expanding representation, and gradually
narrow down (compress toward output). This is called a "funnel" architecture.
Number of hideen layer control the capacity to learn hierarchical feature.

Activation functions :
- ReLU: default for most, fast
- LeakyReLU: use when many neuron dies (output 0).
- Tanh: for centered data (-> 0 mean), slower to train, saturate.
- Sigmoid: only for binary output, easily saturate.
- Softmax: final layer for multi-class classification, convert logic into probabilities.

Type of layer :
- nn.Linear: for tabular data, simple relationship between numerical input.
- nn.Conv2d: for picture data, capture local spatial pattern.
- nn.LSTM, nn.GRU, nn.Transformer: for sequence (text, time), capture temporal and conceptual relationship.
- GCN/GAT: used for graphs/network, learn from node-edge relationship.

How to choose ?
- match data type to layer type.
- if underfitting (model can't capture data complexity, training and validation loss stay high), increase
model width and model depth.
- if overfitting (model is memorizing training data, training loss is ok but validation loss stay
high), decrease model width and model depth.

### Backpropagation

It is how pytorch compute automatically gradients of all parameters.
- 1. forward pass : an input is passed to the model : output = model(input) 
- 2. compute the loss, the loss function measure how wrong model's predictions are compared to
the true target value. Typical functions are mean sqared error for regression task, and
cross entropy loop for classification task.
- 3. backward pass : for each parameters compute partial derivative of the loss
- 4. store gradient.

### Optimizer (parameters update)

Use Adam optimizer.
Once gradient are computed, an optimizer update the parameters.
- learning rate: meta parameters controling the way parameters are updated.
    The gradient tells you which way is downhill. The learning rate tells you how big a step you take in that direction. If your steps are too big you stumble around, if too low you crawl forever.

### Replay buffer

Instead of learning of the action just took, we learn of a random batch of a replay buffer list.
This is used to prevent catastrophic forgeting.
Experience replay make the training task more like supervised learning.
One can collect experience from human gameplay and train the model on these.
- replay_buffer_size: With a large replay buffer, rare transition (eating a green apple) are more
used in training. This add more stability to training.
- replay_buffer_batch_size control how many experiences are sampled from the replay buffer for
each update. A large value means smoother gradient but slow updates (so slow the training process).

### Target model

Similarly to catastrophic forgetting problem mitigated by replay buffer, we often have to mitigate instability
by using a separate target model. This "target network" is a copy of the regular network but lag a little, it is not
updated at every step. This target_network is used to produce the Q values that is used to train the regular model.
Periodically the target network is updated by the parameters of the regular network.
We can see that when the regular model is updated, the loss increase slightly (instability).
- target_model_update_freq: control the frequency of target model update.

### Epsilon Greedy

For training we must sometimes choose random action instead of the best predicted action.
This allow the model to learn. The rate of random action should decrease while training,
allowing the model to perform better and be trained in production performance.
For example the snake need to learn how to manage to not hurt itself when it's big.
epsilon_init, epsilon_min, epsilon_a and epsilon_b are parts of meta-parameters.

## Some papers

- Human level control through deep reinforcement learning (Google Deepmind)
- Neural fitted Qiteration (2005)
- Deep autoencoder NN in RL (2010)

## TODO List

- Calculate epsilon depending on training time
- Train with graphic card, how to ?
- export json file for metrics, to easily compare training/validation.
