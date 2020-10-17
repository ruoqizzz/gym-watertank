from gym.envs.registration import register

register(
    id='watertank-v0',
    entry_point='gym_watertank.envs:WaterTankEnv',
)

register(
    id='watertanklqr-v0',
    entry_point='gym_watertank.envs:WaterTankLQREnv',
)

register(
    id='lqr-v0',
    entry_point='gym_watertank.envs:LQREnv',
)