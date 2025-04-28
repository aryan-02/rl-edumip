from gymnasium.envs.registration import register

register(
    id="ContinuousCartPole-v0",
    entry_point="continuous_cartpole.continuous_cartpole:ContinuousCartPoleEnv",
)
