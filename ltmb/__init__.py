from gymnasium.envs.registration import register

register(
     id='LTMB-Hallway-v0',
     entry_point="ltmb.envs:HallwayEnv",
)

register(
     id='LTMB-Mimic-v0',
     entry_point="ltmb.envs:MimicEnv",
)

register(
     id='LTMB-Counting-v0',
     entry_point="ltmb.envs:CountingEnv",
)