import gymnasium as gym
gym.make(
    'FrozenLake-v1',
    desc=None,
    map_name="4x4",
    is_slippery=True,
    success_rate=1.0/3.0,
    reward_schedule=(1, 0, 0)
)

"4x4":[
    "SFFF",
    "FHFH",
    "FFFH",
    "HFFG"
]