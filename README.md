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
- v5 model is the current best.
- Model meta parameters is passed through yaml, allowing easy experimentation. TODO this is training parameters...

## TODO List

- Calculate epsilon depending on training time
- Document meta paramters and rational value choices
- Add a documentation about DQN and the features of DQN here: target network, replay buffer, ...
- Save loss only once per epoch to optimize a little
- Export metrics to a json file, to easily compare run and run test.
