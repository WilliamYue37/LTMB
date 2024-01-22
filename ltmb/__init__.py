from gymnasium.envs.registration import register

register(
     id='LTMB-Hallway-v0',
     entry_point="ltmb.envs:HallwayEnv",
)

register(
     id='LTMB-Ordering-v0',
     entry_point="ltmb.envs:OrderingEnv",
)

register(
     id='LTMB-Counting-v0',
     entry_point="ltmb.envs:CountingEnv",
)