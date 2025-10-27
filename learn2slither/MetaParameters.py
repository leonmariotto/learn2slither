from pydantic import BaseModel, computed_field

class MetaParameters(BaseModel):
    # The gradient tells you which way is downhill.
    # The learning rate tells you how big a step you take in that direction.
    # If your steps are too big you stumble around, if too low you crawl forever.
    learning_rate: float
    # Gamma meta parameter is applied in Bellman equation, it's the future reward discount factor.
    # It determines how much future rewards matter compared to the current one.
    # The model have the ability to think with many-step plan in mind, as (γ * max(Q_target(S₄))) can propagate
    # to a 5 step away reward. But in this case the 5 step away reward will appear discounted by
    # GAMMA ^ 5 => 0.9 ^ 5 => 0.59
    # With a value of 0 only current reward matter, a value of 0.99 focus strongly on future rewards.
    gamma: float
    # With a large replay buffer, rare transition (eating a green apple) are more used in training.
    # This add more stability to training.
    replay_buffer_size: int
    # Control how many experiences are sampled from the replay buffer for each update. A large value
    # means smoother gradient but slow updates (so slow the training process).
    replay_buffer_batch_size: int
    # Say that 300 step is an epoch
    # 300 step is sufficient to reach a 10-case long snake.
    # Tis define the epsilon update frequency, metrics collecting frequency,
    # and is part of target update frequency.
    # Also currently there is a reset of the game every epoch... TODO is it necessary/authorized/usefull ?
    step_per_epoch: int
    # Frequency of target model update in epoch.
    target_model_update_freq: int
    # Frequency of epsilon update in epoch.
    epsilon_update_freq: int
    # Initial value for epsilon.
    epsilon_init: float
    # Minimum value for epsilon while training.
    epsilon_min: float
    # At each epoch epsilon is computed as: new_epsilon = a * epsilon + b
    # But can't go under epsilon_min.
    epsilon_a: float
    # At each epoch epsilon is computed as: new_epsilon = a * epsilon + b
    # But can't go under epsilon_min.
    epsilon_b: float
