from gym.envs.registration import register

register(
    id='WaterTank-v0',
    entry_point='gym_watertank.envs:WaterTankEnv',
)

register(
    id='WaterTankLQR-v0',
    entry_point='gym_watertank.envs:WaterTankLQREnv',
)

register(
    id='WaterTankLQRZeroNoise-v0',
    entry_point='gym_watertank.envs:WaterTankLQRZeroNoiseEnv',
)

register(
    id='LQR-v0',
    entry_point='gym_watertank.envs:LQREnv',
)