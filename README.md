# Learn2Slither

## HowTo

Use uv to pull necessary packages.
```
uv sync
```
Run trained model.
```
uv run agent.py -mp meta/v5.yml -i models/v5.weight --play
```

If you want to play snake just run `uv run play_snake.py`.

## Training

Run `uv run agent.py --help` for a list of available option.
A model can be loaded using `-i` option, the trained model can be output
using `-o` option.
To re-train a model one can use :
```
uv run agent.py -i v1.weight -o v1.weight -t 600
```
The `-t` option specify the time of training (in second, so 10mn in this case).

## Model

- This is a Deep Q Network

The model take the state of Snake in input and output a Q value prediction of action-rewards.</br>
There is an internal node layer which take a vector of dimension 150 and ouput a vector of 
dimension 100.
```
NN_L1 = 24
NN_L2 = 150
NN_L3 = 100
NN_L4 = 4
self.model = torch.nn.Sequential(
    torch.nn.Linear(NN_L1, NN_L2),
    torch.nn.ReLU(),
    torch.nn.Linear(NN_L2, NN_L3),
    torch.nn.ReLU(),
    torch.nn.Linear(NN_L3, NN_L4),
)
```

### Replay buffer
TODO
### Target model
TODO
### Optimizer
TODO
### Epsilon Greedy
TODO
### Loss function
TODO

- v5 model is the current best.
- Model meta parameters is passed through yaml, allowing easy experimentation. TODO this is training parameters...

## TODO List

- Calculate epsilon depending on training time
- Document meta paramters and rational value choices
- Add a documentation about DQN and the features of DQN here: target network, replay buffer, ...
- Save loss only once per epoch to optimize a little
- Export metrics to a json file, to easily compare run and run test.
