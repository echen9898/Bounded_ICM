from gym.envs.registration import register

register(
    id='VizdoomBasic-v0',
    entry_point='vizdoomgym.envs:VizdoomBasic',
    max_episode_steps=10000,
    reward_threshold=10.0 
)

register(
    id='VizdoomCorridor-v0',
    entry_point='vizdoomgym.envs:VizdoomCorridor',
    max_episode_steps=10000,
    reward_threshold=1000.0
)

register(
    id='VizdoomDefendCenter-v0',
    entry_point='vizdoomgym.envs:VizdoomDefendCenter',
    max_episode_steps=10000,
    reward_threshold=10.0
)

register(
    id='VizdoomDefendLine-v0',
    entry_point='vizdoomgym.envs:VizdoomDefendLine',
    max_episode_steps=10000,
    reward_threshold=15.0
)

register(
    id='VizdoomHealthGathering-v0',
    entry_point='vizdoomgym.envs:VizdoomHealthGathering',
    max_episode_steps=10000,
    reward_threshold=1000.0
)

register(
    id='VizdoomMyWayHome-v0',
    entry_point='vizdoomgym.envs:VizdoomMyWayHome',
    max_episode_steps=2099, # This value must be one less than the episode_timeout value set in the .cfg file
    reward_threshold=0.5
)

register(
    id='VizdoomMyWayHomeFixed-v0',
    entry_point='vizdoomgym.envs:VizdoomMyWayHomeFixedEnv',
    max_episode_steps=2099, # This value must be one less than the episode_timeout value set in the .cfg file
    reward_threshold=0.5
)

register(
    id='VizdoomMyWayHomeFixed15-v0',
    entry_point='vizdoomgym.envs:VizdoomMyWayHomeFixed15Env',
    max_episode_steps=2099, # This value must be one less than the episode_timeout value set in the .cfg file
    reward_threshold=0.5
)

register(
    id='VizdoomPredictPosition-v0',
    entry_point='vizdoomgym.envs:VizdoomPredictPosition',
    max_episode_steps=10000,
    reward_threshold=0.5
)

register(
    id='VizdoomTakeCover-v0',
    entry_point='vizdoomgym.envs:VizdoomTakeCover',
    max_episode_steps=10000,
    reward_threshold=750.0
)

register(
    id='VizdoomDeathmatch-v0',
    entry_point='vizdoomgym.envs:VizdoomDeathmatch',
    max_episode_steps=10000,
    reward_threshold=20.0
)

register(
    id='VizdoomHealthGatheringSupreme-v0',
    entry_point='vizdoomgym.envs:VizdoomHealthGatheringSupreme',
    max_episode_steps=10000,
)

register(
    id='VizdoomLabyrinthSingle-v0',
    entry_point='vizdoomgym.envs:VizdoomLabyrinthSingle',
    max_episode_steps=2099, # This value must be one less than the episode_timeout value set in the .cfg file
    reward_threshold=0.5
)

register(
    id='VizdoomLabyrinthMany-v0',
    entry_point='vizdoomgym.envs:VizdoomLabyrinthMany',
    max_episode_steps=2099, # This value must be one less than the episode_timeout value set in the .cfg file
    reward_threshold=0.5
)

register(
    id='VizdoomLabyrinthManyFixed-v0',
    entry_point='vizdoomgym.envs:VizdoomLabyrinthManyFixed',
    max_episode_steps=2099, # This value must be one less than the episode_timeout value set in the .cfg file
    reward_threshold=0.5
)

register(
    id='VizdoomLabyrinthManyFixedAngle-v0',
    entry_point='vizdoomgym.envs:VizdoomLabyrinthManyFixedAngle',
    max_episode_steps=2099, # This value must be one less than the episode_timeout value set in the .cfg file
    reward_threshold=0.5
)

