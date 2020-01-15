#!/usr/bin/env python
from __future__ import print_function
import gym
import numpy as np
import argparse
import logging
from pyvirtualdisplay import Display
from envs import create_env
import tensorflow as tf
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def inference(args):
    """
    It restore policy weights, and does inference.
    """
    # virtual display (headless remotes)
    virtual_display = Display(visible=0, size=(1400, 900))
    virtual_display.start()

    # Observation normalization
    obs_mean = None
    obs_std = None
    if args.obs_norm:
        tmp_env = create_env(args.env_id, client_id='0', remotes=None, envWrap=True, acRepeat=1,
                            record=False, record_frequency=1, outdir=None)
        observations = list()
        _ = tmp_env.reset()
        for _ in range(1000): # collect 10000 random observations
            stepAct = np.random.randint(0, tmp_env.action_space.n) # random actions
            state, _, terminal, _ = tmp_env.step(stepAct)
            observations.append(state)
            if terminal:
                tmp_env.reset()
        obs_mean = np.mean(observations, axis=0)
        obs_std = np.std(observations, axis=0)
        tmp_env.close()

    # define environment
    env = create_env(args.env_id, client_id='0', remotes=None, envWrap=True,
                        acRepeat=1, record=args.record, outdir=args.outdir)
    num_actions = env.action_space.n

    with tf.device("/cpu:0"):
        config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
        with tf.Session(config=config) as sess:
            logger.info("Restoring trainable global parameters.")
            saver = tf.train.import_meta_graph(args.ckpt+'.meta', clear_devices=True) # ADDED CLEAR DEVICES
            saver.restore(sess, args.ckpt)

            # Collections are saved in model.py under LSTM policy
            probs = tf.get_collection("probs")[0]
            sample = tf.get_collection("sample")[0]
            vf = tf.get_collection("vf")[0]
            state_out_0 = tf.get_collection("state_out_0")[0]
            state_out_1 = tf.get_collection("state_out_1")[0]

            last_state = env.reset()

            # normalize observation if needed
            if obs_mean is not None and obs_std is not None:
                last_state = (last_state-obs_mean)/obs_std

            if args.render or args.record:
                env.render()
            last_features = np.zeros((1, 256), np.float32); last_features = [last_features, last_features]
            length = 0
            rewards = 0
            mario_distances = np.zeros((args.num_episodes,))
            for i in range(args.num_episodes):
                print("Starting episode %d" % (i + 1))
                if args.random:
                    print('I am a random policy!')
                else:
                    if args.greedy:
                        print('I am a greedy policy!')
                    else:
                        print('I am a sampled policy!')
                while True:
                    # run policy
                    fetched = sess.run([probs, sample, vf, state_out_0, state_out_1] ,
                                {"global/x:0": [last_state], "global/c_in:0": last_features[0], "global/h_in:0": last_features[1]})
                    prob_action, action, value_, features = fetched[0], fetched[1], fetched[2], fetched[3:]

                    # run environment
                    if args.random:
                        stepAct = np.random.randint(0, num_actions)  # random policy
                    else:
                        if args.greedy:
                            stepAct = prob_action.argmax()  # greedy policy
                        else:
                            stepAct = action.argmax()
                    state, reward, terminal, info = env.step(stepAct)

                    # normalize observations if needed
                    if obs_mean is not None and obs_std is not None:
                        state = (state-obs_mean)/obs_std

                    # update stats
                    length += 1
                    rewards += reward
                    last_state = state
                    last_features = features
                    if args.render or args.record:
                        env.render(mode='rgb_array') # set to rgb_array by default (assumes running on a headless remote)

                    timestep_limit = env.spec.tags.get('wrapper_config.TimeLimit.max_episode_steps')
                    if timestep_limit is None: timestep_limit = env.spec.timestep_limit
                    if terminal or length >= timestep_limit:
                        if length >= timestep_limit or not env.metadata.get('semantics.autoreset'):
                            last_state = env.reset()
                        last_features = np.zeros((1, 256), np.float32); last_features = [last_features, last_features]
                        print("Episode finished. Sum of rewards: %.2f. Length: %d." % (rewards, length))
                        length = 0
                        rewards = 0
                        if args.render or args.record:
                            env.render(mode='rgb_array')
                        break

    logger.info('Finished %d true episodes.', args.num_episodes)
    env.close()


def main(_):
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('--ckpt', default="../models/doom/doom_ICM", help='checkpoint name')
    parser.add_argument('--outdir', default="../models/output", help='Output log directory')
    parser.add_argument('--env-id', default="doom", help='Environment id')
    parser.add_argument('--record', action='store_true', help="Record the policy running video")
    parser.add_argument('--render', action='store_true', help="Render the gym environment video online")
    parser.add_argument('--num-episodes', type=int, default=2, help="Number of episodes to run")
    parser.add_argument('--greedy', action='store_true', help="Default sampled policy. This option does argmax")
    parser.add_argument('--random', action='store_true', help="Default sampled policy. This option does random policy")
    parser.add_argument('--obs-norm', action='store_true', help="Whether or not you should normalize the observations")
    parser.add_argument('--demo', action='store_true', help='Whether or not youre using the demo model provided by the authors')
    args = parser.parse_args()
    inference(args)

if __name__ == "__main__":
    tf.app.run()
