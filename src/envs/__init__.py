from gymnasium.envs.registration import register

register(
    id='GenesisLung-v0',
    entry_point='src.envs.lung_env:LungEnv',
)
