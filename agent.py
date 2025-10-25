#!/usr/bin/env python3

import time

from snake.Snake import Snake
from snake.Agent import Agent
import click

FPS = 60
MOVE_PER_SECOND = 6

def play_with_agent(agent):
    snake = Snake(render=True)
    last_render = time.perf_counter()
    last_move = time.perf_counter()
    while True:
        now = time.perf_counter()
        if (now - last_move >= 1 / MOVE_PER_SECOND):
            snake.poll_input() # catch window closure.
            agent.run_best_step(snake)
            last_move = now
        if (now - last_render >= 1 / 60):
            snake.draw()
            last_render = now


@click.command()
@click.option(
    "--time-limit",
    "-t",
    type=int,
    default=60,
    show_default=True,
    help="Time of play, in second.",
)
@click.option(
    "--out_weight",
    "-o",
    type=str,
    default="",
    help="Filepath to output weigth file.",
)
@click.option(
    "--in_weight",
    "-i",
    type=str,
    default="",
    help="Filepath to input weigth file.",
)
@click.option(
    "--play",
    is_flag=True,
    default=False,
    show_default=True,
    help="Play with the agent, display it.",
)
def run_agent(
    time_limit: int,
    out_weight: str,
    in_weight: str,
    play: bool
):
    agent = Agent()
    if in_weight != "":
        agent.import_weight(in_weight)

    if play:
        play_with_agent(agent)
    else:
        snake = Snake(render=False)
        start = time.perf_counter()
        while time.perf_counter() - start < time_limit:
            agent.run_step(snake)
        if out_weight != "":
            agent.export_weight(out_weight)
        print(f"Final epsilon = {agent.epsilon}")
        print(f"Epoch numbers = {agent.epoch_counter}")
        agent.plot_losses()
        agent.plot_rewards()
        agent.plot_epsilons()
        agent.show()

if __name__ == "__main__":
    run_agent()
