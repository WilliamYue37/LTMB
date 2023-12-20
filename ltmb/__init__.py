from gymnasium.envs.registration import register

register(
     id='LTMB-Hallway-v0',
     entry_point="ltmb.envs:HallwayEnv",
)