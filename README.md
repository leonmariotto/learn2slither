# Learn2Slither

## HowTo

Create and activate virtual env.
```
python3 -m venv .venv
. .venv/bin/activate
```

Install requirements.
```
pip install -r requirements.txt
```

Run trained model.
```
./agent.py -i v1.weight --play
```

If you want to play snake just run `./play_snake.py`.

## Training

Run `agent.py --help` for a list of available option.
A model can be loaded using `-i` option, the trained model can be output
using `-o` option.
To re-train a model one can use :
```
./agent.py -i v1.weight -o v1.weight -t 600
```
The `-t` option specify the time of training (in second, so 10mn in this case).

## Model

This is a Deep Q Network.
