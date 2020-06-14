from __future__ import print_function
import go_vncdriver
import tensorflow as tf
import numpy as np
import argparse
import logging
import os
import gym
from envs import create_env
from model import LSTMPolicy, StatePredictor, StateActionPredictor
import utils
import distutils.version
from pyvirtualdisplay import Display
use_tf12_api = distutils.version.LooseVersion(tf.VERSION) >= distutils.version.LooseVersion('0.12.0')

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Disables write_meta_graph argument, which freezes entire process and is mostly useless.
class FastSaver(tf.train.Saver):
    def save(self, sess, save_path, global_step=None, latest_filename=None,
             meta_graph_suffix="meta", write_meta_graph=True):
        super(FastSaver, self).save(sess, save_path, global_step, latest_filename,
                                    meta_graph_suffix, False)

# Used to run an inference process during training
class InferenceAgent(object):
    def __init__(self, env, task, visualise, unsupType, designHead='universe'):
        self.env = env
        self.task = task
        self.visualise = visualise
        self.unsupType = unsupType
        self.designHed = designHead

        numaction = env.action_space.n
        worker_device = "/job:worker/task:{}/cpu:0".format(self.task)

        with tf.device(tf.train.replica_device_setter(1, worker_device=worker_device)):
            with tf.variable_scope("global"):
                self.network = LSTMPolicy(env.observation_space.shape, numaction, designHead)
                self.global_step = tf.get_variable("global_step", [], tf.int32, initializer=tf.constant_initializer(0, dtype=tf.int32),
                                                   trainable=False)
                if self.unsupType is not None:
                    with tf.variable_scope("predictor"):
                        if 'state' in self.unsupType:
                            self.ap_network = StatePredictor(env.observation_space.shape, numaction, designHead, unsupType)
                        else:
                            self.ap_network = StateActionPredictor(env.observation_space.shape, numaction, designHead)

        with tf.device(worker_device):
            with tf.variable_scope("local"):
                self.local_network = LSTMPolicy(env.observation_space.shape, numaction, designHead)
                self.local_network.global_step = self.global_step
                sync_var_list = [v1.assign(v2) for v1, v2 in zip(self.local_network.var_list, self.network.var_list)]
                self.sync = tf.group(*sync_var_list)

    def run_inference(self, sess, env, summary_writer):
        self.sess = sess

        with self.sess.as_default():

            self.sess.run(self.sync) # pull global parameters
            print("\n")
            print("PULLING PARAMS")
            print("\n")
            last_state = env.reset()
            if self.visualise:
                env.render()
            length = 0
            rewards = 0
            last_features = self.local_network.get_initial_features()  # reset lstm memory

            for _ in range(10): # gather 10 inference runs for each set of global parameters

                while True:
                    # run policy
                    fetched = self.local_network.act_inference(last_state, *last_features)
                    prob_action, action, value_, features = fetched[0], fetched[1], fetched[2], fetched[3:]

                    # run environment: sampled one-hot 'action' (not greedy)
                    stepAct = action.argmax()

                    # print(stepAct, prob_action.argmax(), prob_action)
                    state, reward, terminal, info = env.step(stepAct)

                    # update stats
                    length += 1
                    rewards += reward
                    last_state = state
                    last_features = features
                    if self.visualise:
                        env.render()

                    # store relevant summaries
                    timestep_limit = env.spec.tags.get('wrapper_config.TimeLimit.max_episode_steps')
                    if timestep_limit is None: timestep_limit = env.spec.timestep_limit
                    
                    if terminal or length >= timestep_limit:
                        summary = tf.Summary()
                        if 'distance' in info:
                            summary.value.add(tag='inference_distance', simple_value=info['distance'])
                        summary_writer.add_summary(summary, self.local_network.global_step.eval())
                        summary_writer.flush()

                        if length >= timestep_limit or not env.metadata.get('semantics.autoreset'):
                            last_state = env.reset()
                        last_features = self.local_network.get_initial_features()  # reset lstm memory
                        print("Episode finished. Sum of rewards: %.2f. Length: %d." % (rewards, length))
                        if 'distance' in info:
                            print('Mario Distance Covered:', info['distance'])
                        length = 0
                        rewards = 0
                        if self.visualise:
                            env.render()
                        break


# Used to run inference post training
def inference(args):
    """
    It only restores LSTMPolicy architecture, and does inference using that.
    """
    # get address of checkpoints
    indir = os.path.join(args.log_dir, 'train')
    outdir = os.path.join(args.log_dir, 'inference') if args.out_dir is None else args.out_dir
    with open(indir + '/checkpoint', 'r') as f:
        first_line = f.readline().strip()
    ckpt = first_line.split(' ')[-1].split('/')[-1][:-1]
    ckpt = ckpt.split('-')[-1]
    ckpt = indir + '/model.ckpt-' + ckpt

    virtual_display = Display(visible=0, size=(1400, 900))
    virtual_display.start()

    # define environment
    if args.record:
        env = create_env(args.env_id, client_id='0', remotes=None, envWrap=args.envWrap, designHead=args.designHead,
                            record=True, noop=args.noop, acRepeat=args.acRepeat, outdir=outdir, record_frequency=1)
    else:
        env = create_env(args.env_id, client_id='0', remotes=None, envWrap=args.envWrap, designHead=args.designHead,
                            record=True, noop=args.noop, acRepeat=args.acRepeat)
    numaction = env.action_space.n

    with tf.device("/cpu:0"):
        # define policy network
        with tf.variable_scope("global"):
            policy = LSTMPolicy(env.observation_space.shape, numaction, args.designHead)
            policy.global_step = tf.get_variable("global_step", [], tf.int32, initializer=tf.constant_initializer(0, dtype=tf.int32),
                                               trainable=False)

        # Variable names that start with "local" are not saved in checkpoints.
        if use_tf12_api:
            variables_to_restore = [v for v in tf.global_variables() if not v.name.startswith("local")]
            init_all_op = tf.global_variables_initializer()
        else:
            variables_to_restore = [v for v in tf.all_variables() if not v.name.startswith("local")]
            init_all_op = tf.initialize_all_variables()
        saver = FastSaver(variables_to_restore)

        # print trainable variables
        var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, tf.get_variable_scope().name)
        logger.info('Trainable vars:')
        for v in var_list:
            logger.info('  %s %s', v.name, v.get_shape())

        # summary of rewards
        action_writers = []
        if use_tf12_api:
            summary_writer = tf.summary.FileWriter(outdir)
            for ac_id in range(numaction):
                action_writers.append(tf.summary.FileWriter(os.path.join(outdir,'action_{}'.format(ac_id))))
        else:
            summary_writer = tf.train.SummaryWriter(outdir)
            for ac_id in range(numaction):
                action_writers.append(tf.train.SummaryWriter(os.path.join(outdir,'action_{}'.format(ac_id))))
        logger.info("Inference events directory: %s", outdir)

        config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
        with tf.Session(config=config) as sess:
            logger.info("Initializing all parameters.")
            sess.run(init_all_op)
            logger.info("Restoring trainable global parameters.")
            saver.restore(sess, ckpt)
            logger.info("Restored model was trained for %.2fM global steps", sess.run(policy.global_step)/1000000.)
            # saving with meta graph:
            # metaSaver = tf.train.Saver(variables_to_restore)
            # metaSaver.save(sess, 'models/doomICM')

            last_state = env.reset()
            if args.render or args.record:
                env.render()
            last_features = policy.get_initial_features()  # reset lstm memory
            length = 0
            rewards = 0
            mario_distances = np.zeros((args.num_episodes,))
            for i in range(args.num_episodes):
                print("Starting episode %d" % (i + 1))
                if args.recordSignal:
                    from PIL import Image
                    signalCount = 1
                    utils.mkdir_p(outdir + '/recordedSignal/ep_%02d/'%i)
                    Image.fromarray((255*last_state[..., -1]).astype('uint8')).save(outdir + '/recordedSignal/ep_%02d/%06d.jpg'%(i,signalCount))

                if args.random:
                    print('I am random policy!')
                else:
                    if args.greedy:
                        print('I am greedy policy!')
                    else:
                        print('I am sampled policy!')
                while True:
                    # run policy
                    fetched = policy.act_inference(last_state, *last_features)
                    prob_action, action, value_, features = fetched[0], fetched[1], fetched[2], fetched[3:]

                    # run environment: sampled one-hot 'action' (not greedy)
                    if args.random:
                        stepAct = np.random.randint(0, numaction)  # random policy
                    else:
                        if args.greedy:
                            stepAct = prob_action.argmax()  # greedy policy
                        else:
                            stepAct = action.argmax()
                    # print(stepAct, prob_action.argmax(), prob_action)
                    state, reward, terminal, info = env.step(stepAct)

                    # update stats
                    length += 1
                    rewards += reward
                    last_state = state
                    last_features = features
                    if args.render or args.record:
                        env.render()
                    if args.recordSignal:
                        signalCount += 1
                        Image.fromarray((255*last_state[..., -1]).astype('uint8')).save(outdir + '/recordedSignal/ep_%02d/%06d.jpg'%(i,signalCount))

                    # store summary
                    summary = tf.Summary()
                    summary.value.add(tag='ep_{}/reward'.format(i), simple_value=reward)
                    summary.value.add(tag='ep_{}/netreward'.format(i), simple_value=rewards)
                    summary.value.add(tag='ep_{}/value'.format(i), simple_value=float(value_[0]))
                    if 'NoFrameskip-v' in args.env_id:  # atari
                        summary.value.add(tag='ep_{}/lives'.format(i), simple_value=env.unwrapped.ale.lives())
                    summary_writer.add_summary(summary, length)
                    summary_writer.flush()
                    summary = tf.Summary()
                    for ac_id in range(numaction):
                        summary.value.add(tag='action_prob', simple_value=float(prob_action[ac_id]))
                        action_writers[ac_id].add_summary(summary, length)
                        action_writers[ac_id].flush()

                    timestep_limit = env.spec.tags.get('wrapper_config.TimeLimit.max_episode_steps')
                    if timestep_limit is None: timestep_limit = env.spec.timestep_limit
                    if terminal or length >= timestep_limit:
                        if length >= timestep_limit or not env.metadata.get('semantics.autoreset'):
                            last_state = env.reset()
                        last_features = policy.get_initial_features()  # reset lstm memory
                        print("Episode finished. Sum of rewards: %.2f. Length: %d." % (rewards, length))
                        if 'distance' in info:
                            print('Mario Distance Covered:', info['distance'])
                            mario_distances[i] = info['distance']
                        length = 0
                        rewards = 0
                        if args.render or args.record:
                            env.render()
                        if args.recordSignal:
                            signalCount += 1
                            Image.fromarray((255*last_state[..., -1]).astype('uint8')).save(outdir + '/recordedSignal/ep_%02d/%06d.jpg'%(i,signalCount))
                        break

        logger.info('Finished %d true episodes.', args.num_episodes)
        if 'distance' in info:
            print('Mario Distances:', mario_distances)
            print('Mario Distance Mean: ', np.mean(mario_distances))
            print('Mario Distance Std: ', np.std(mario_distances))
            np.save(outdir + '/distances.npy', mario_distances)
        env.close()


def main(_):
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('--log-dir', default="tmp/doom", help='input model directory')
    parser.add_argument('--out-dir', default=None, help='output log directory. Default: log_dir/inference/')
    parser.add_argument('--env-id', default="PongDeterministic-v3", help='Environment id')
    parser.add_argument('--record', action='store_true', help="Record the gym environment video -- user friendly")
    parser.add_argument('--recordSignal', action='store_true', help="Record images of true processed input to network")
    parser.add_argument('--render', action='store_true', help="Render the gym environment video online")
    parser.add_argument('--envWrap', action='store_true', help="Preprocess input in env_wrapper (no change in input size or network)")
    parser.add_argument('--designHead', type=str, default='universe', help="Network deign head: nips or nature or doom or universe(default)")
    parser.add_argument('--num-episodes', type=int, default=2, help="Number of episodes to run")
    parser.add_argument('--noop', action='store_true', help="Add 30-noop for inference too (recommended by Nature paper, don't know?)")
    parser.add_argument('--acRepeat', type=int, default=0, help="Actions to be repeated at inference. 0 means default. applies iff envWrap is True.")
    parser.add_argument('--greedy', action='store_true', help="Default sampled policy. This option does argmax.")
    parser.add_argument('--random', action='store_true', help="Default sampled policy. This option does random policy.")
    parser.add_argument('--default', action='store_true', help="run with default params")
    args = parser.parse_args()
    if args.default:
        args.envWrap = True
        args.acRepeat = 1
    if args.acRepeat <= 0:
        print('Using default action repeat (i.e. 4). Min value that can be set is 1.')

    inference(args)

if __name__ == "__main__":
    tf.app.run()
