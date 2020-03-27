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
    max_episode_steps=2099, # See above
    reward_threshold=0.5
)

register(
    id='VizdoomMyWayHomeFixed15-v0',
    entry_point='vizdoomgym.envs:VizdoomMyWayHomeFixed15Env',
    max_episode_steps=2099, # See above
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
    id='LabyrinthSingle-v0',
    entry_point='vizdoomgym.envs:VizdoomLabyrinthSingle',
    max_episode_steps=2099,
    reward_threshold=0.5
)

register(
    id='LabyrinthMany-v0',
    entry_point='vizdoomgym.envs:VizdoomLabyrinthMany',
    max_episode_steps=2099,
    reward_threshold=0.5
)

register(
    id='LabyrinthManyFixed-v0',
    entry_point='vizdoomgym.envs:VizdoomLabyrinthManyFixed',
    max_episode_steps=2099,
    reward_threshold=0.5
)

# -------------------------------------------------------------------- LABYRINTH ENVIRONMENTS --------------------------------------------------------------------
register(
    id='LabyrinthRandTx-1-v0',
    entry_point='vizdoomgym.envs:VizdoomLabyrinthRandTx_1',
    max_episode_steps=2099,
    reward_threshold=0.5
)

register(
    id='LabyrinthRandTx-2-v0',
    entry_point='vizdoomgym.envs:VizdoomLabyrinthRandTx_2',
    max_episode_steps=2099,
    reward_threshold=0.5
)

register(
    id='LabyrinthRandTx-3-v0',
    entry_point='vizdoomgym.envs:VizdoomLabyrinthRandTx_3',
    max_episode_steps=2099,
    reward_threshold=0.5
)

register(
    id='LabyrinthRandTx-4-v0',
    entry_point='vizdoomgym.envs:VizdoomLabyrinthRandTx_4',
    max_episode_steps=2099,
    reward_threshold=0.5
)

register(
    id='LabyrinthRandTx-5-v0',
    entry_point='vizdoomgym.envs:VizdoomLabyrinthRandTx_5',
    max_episode_steps=2099,
    reward_threshold=0.5
)

register(
    id='LabyrinthRandTx-6-v0',
    entry_point='vizdoomgym.envs:VizdoomLabyrinthRandTx_6',
    max_episode_steps=2099,
    reward_threshold=0.5
)

register(
    id='LabyrinthRandTx-7-v0',
    entry_point='vizdoomgym.envs:VizdoomLabyrinthRandTx_7',
    max_episode_steps=2099,
    reward_threshold=0.5
)

register(
    id='LabyrinthRandTx-8-v0',
    entry_point='vizdoomgym.envs:VizdoomLabyrinthRandTx_8',
    max_episode_steps=2099,
    reward_threshold=0.5
)

register(
    id='LabyrinthRandTx-9-v0',
    entry_point='vizdoomgym.envs:VizdoomLabyrinthRandTx_9',
    max_episode_steps=2099,
    reward_threshold=0.5
)

register(
    id='LabyrinthRandTx-10-v0',
    entry_point='vizdoomgym.envs:VizdoomLabyrinthRandTx_10',
    max_episode_steps=2099,
    reward_threshold=0.5
)

register(
    id='LabyrinthRandTx-11-v0',
    entry_point='vizdoomgym.envs:VizdoomLabyrinthRandTx_11',
    max_episode_steps=2099,
    reward_threshold=0.5
)

register(
    id='LabyrinthRandTx-12-v0',
    entry_point='vizdoomgym.envs:VizdoomLabyrinthRandTx_12',
    max_episode_steps=2099,
    reward_threshold=0.5
)

register(
    id='LabyrinthRandTx-13-v0',
    entry_point='vizdoomgym.envs:VizdoomLabyrinthRandTx_13',
    max_episode_steps=2099,
    reward_threshold=0.5
)

register(
    id='LabyrinthRandTx-14-v0',
    entry_point='vizdoomgym.envs:VizdoomLabyrinthRandTx_14',
    max_episode_steps=2099,
    reward_threshold=0.5
)

register(
    id='LabyrinthRandTx-15-v0',
    entry_point='vizdoomgym.envs:VizdoomLabyrinthRandTx_15',
    max_episode_steps=2099,
    reward_threshold=0.5
)

register(
    id='LabyrinthRandTx-16-v0',
    entry_point='vizdoomgym.envs:VizdoomLabyrinthRandTx_16',
    max_episode_steps=2099,
    reward_threshold=0.5
)

register(
    id='LabyrinthRandTx-17-v0',
    entry_point='vizdoomgym.envs:VizdoomLabyrinthRandTx_17',
    max_episode_steps=2099,
    reward_threshold=0.5
)

register(
    id='LabyrinthRandTx-18-v0',
    entry_point='vizdoomgym.envs:VizdoomLabyrinthRandTx_18',
    max_episode_steps=2099,
    reward_threshold=0.5
)

register(
    id='LabyrinthRandTx-19-v0',
    entry_point='vizdoomgym.envs:VizdoomLabyrinthRandTx_19',
    max_episode_steps=2099,
    reward_threshold=0.5
)

register(
    id='LabyrinthRandTx-20-v0',
    entry_point='vizdoomgym.envs:VizdoomLabyrinthRandTx_20',
    max_episode_steps=2099,
    reward_threshold=0.5
)