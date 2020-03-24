from __future__ import print_function
from collections import namedtuple
import numpy as np
import tensorflow as tf
from model import LSTMPolicy, StateActionPredictor, StatePredictor
import six.moves.queue as queue
import scipy.signal
import threading
import distutils.version
from constants import constants
from utils import RunningMeanStd, update_mean_var_count_from_moments
use_tf12_api = distutils.version.LooseVersion(tf.VERSION) >= distutils.version.LooseVersion('0.12.0')
import pdb


def discount(x, gamma, trivial=False):
    """
        x = [r1, r2, r3, ..., rN]
        returns [r1 + r2*gamma + r3*gamma^2 + ...,
                   r2 + r3*gamma + r4*gamma^2 + ...,
                     r3 + r4*gamma + r5*gamma^2 + ...,
                        ..., ..., rN]

        if trivial mode is on, then instead returns:
            [r1 + r2*gamma + r3*gamma^2 + ... ] only.
    """
    if trivial == True:
        return scipy.signal.lfilter([1], [1, -gamma], x[::-1], axis=0)[::-1][0]
    return scipy.signal.lfilter([1], [1, -gamma], x[::-1], axis=0)[::-1]

def zero_pad_multistep_bonuses(bonuses, horizon):
    """
    Takes in [ [r_1, r_2 ... r_k], ... [r_1, r_2, ... , None, None, None] ] rollouts corresponding to each (s_t, a_t), 
    and replaces all Nones with zeroes the unobtainable horizon predictions at the end of the episodes.
    """
    for h in range(horizon):
        for i in range(len(bonuses[-(h+1)])):
            if bonuses[-(h+1)][i] is None:
                bonuses[-(h+1)][i] = 0.0
    return bonuses

def group_action_sequences(actions, h):
    """
    Takes in a sequence of actions, and groups them in terms of horizon h rollouts starting from each action:
    [a_1, a_2, a_3], h=2  --->  [[a_1 + a_2], [a_2 + a_3]]
    """
    grouped = list()
    for i in range(len(actions)-h+1):
        action_seq = actions[i:i+h]
        grouped.append(np.concatenate(action_seq))
    return grouped

def process_rollout(rollout, gamma, lambda_=1.0, clip=False, adv_norm=False, r_std_running=False, backup_bound=None, horizon=1, mstep_mode='sum'):
    """
    Given a rollout, compute its returns and the advantage.
    """

    # Zero out all Nones for the multistep predictions
    rollout.multistep_bonuses = zero_pad_multistep_bonuses(rollout.multistep_bonuses, horizon)

    # Processes multistep predictions
    if horizon == 1: # standard ICM
        rollout.bonuses = np.concatenate(rollout.multistep_bonuses)
    else:  # multistep prediction
        if mstep_mode == 'sum':
            rollout.bonuses = np.sum(rollout.multistep_bonuses, axis=1) # max, average, discounted sum, whatever you want happens here.
        elif mstep_mode == 'dissum':
            rollout.bonuses = np.apply_along_axis(discount, 1, rollout.multistep_bonuses, gamma=constants['MULTISTEP_GAMMA'], trivial=True)
        elif mstep_mode == 'max':
            rollout.bonuses = np.max(rollout.multistep_bonuses, axis=1)

    # collecting state transitions
    if rollout.unsup:
        batch_si = np.asarray(rollout.states + [rollout.end_state])
    else:
        batch_si = np.asarray(rollout.states)

    # grouping action sequences (for multi-step prediction)
    batch_a = dict()
    for h in range(horizon):
        batch_a[h] = group_action_sequences(rollout.actions, h+1)
    
    # Normalize rewards
    rewards = np.asarray(rollout.rewards)
    if r_std_running:
        r_mean = np.mean(rewards)
        r_std = np.std(rewards)
        r_std_running.update_from_moments(r_mean, r_std**2, len(rewards))
        rewards = rewards / np.sqrt(r_std_running.var)

    # collecting target for value network
    # V_t <-> r_t + gamma*r_{t+1} + ... + gamma^n*r_{t+n} + gamma^{n+1}*V_{n+1}
    rewards_plus_v = np.asarray(list(rewards) + [rollout.r])  # bootstrapping
    if rollout.unsup:
        rewards_plus_v += np.asarray(list(rollout.bonuses) + [0])
    if clip:
        rewards_plus_v[:-1] = np.clip(rewards_plus_v[:-1], -constants['REWARD_CLIP'], constants['REWARD_CLIP'])
    batch_r = discount(rewards_plus_v, gamma)[:-1]  # value network target

    # collecting target for policy network
    if rollout.unsup: rewards += np.asarray(rollout.bonuses)
    if clip: rewards = np.clip(rewards, -constants['REWARD_CLIP'], constants['REWARD_CLIP'])
    vpred_t = np.asarray(rollout.values + [rollout.r])

    # "Generalized Advantage Estimation": https://arxiv.org/abs/1506.02438
    # Eq (10): delta_t = Rt + gamma*V_{t+1} - V_t
    # Eq (16): batch_adv_t = delta_t + (gamma*lambda)delta_{t+1} + (gamma*lambda)^2*delta_{t+2} + ...
    delta_t = rewards + gamma * vpred_t[1:] - vpred_t[:-1]
    batch_adv = discount(delta_t, gamma * lambda_)

    # Bound the advantage
    if backup_bound != -1.0:
        batch_adv[np.where(vpred_t[:-1]>float(backup_bound))] = np.array([0.0])

    # Normalize batch advantage
    if adv_norm: batch_adv_normed = (batch_adv - np.mean(batch_adv))/(np.std(batch_adv) + 1e-7)

    features = rollout.features[0]

    if adv_norm: return Batch(batch_si, batch_a, batch_adv_normed, batch_r, r_std_running, rollout.terminal, features, vpred_t[:-1])
    else: return Batch(batch_si, batch_a, batch_adv, batch_r, r_std_running, rollout.terminal, features, vpred_t[:-1])

Batch = namedtuple("Batch", ["si", "a", "adv", "r", "r_std_running", "terminal", "features", "vpreds"])

class PartialRollout(object):
    """
    A piece of a complete rollout.  We run our agent, and process its experience
    once it has processed enough steps.
    """
    def __init__(self, unsup=False, horizon = 1):
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.r = 0.0
        self.terminal = False
        self.features = []
        self.unsup = unsup
        if self.unsup:
            self.bonuses = []
            self.end_state = None

    def add(self, state, action, reward, value, terminal, features,
                bonus=None, end_state=None, multistep_bonuses=None):
        self.states += [state]
        self.actions += [action]
        self.rewards += [reward]
        self.values += [value]
        self.terminal = terminal
        self.features += [features]
        if self.unsup:
            self.bonuses += [bonus]
            self.end_state = end_state
            self.multistep_bonuses = multistep_bonuses

    def extend(self, other):
        assert not self.terminal
        self.states.extend(other.states)
        self.actions.extend(other.actions)
        self.rewards.extend(other.rewards)
        self.values.extend(other.values)
        self.r = other.r
        self.terminal = other.terminal
        self.features.extend(other.features)
        if self.unsup:
            self.bonuses.extend(other.bonuses)
            self.end_state = other.end_state
            self.multistep_bonuses.extend(other.multistep_bonuses)

class RunnerThread(threading.Thread):
    """
    One of the key distinctions between a normal environment and a universe environment
    is that a universe environment is _real time_.  This means that there should be a thread
    that would constantly interact with the environment and tell it what to do.  This thread is here.
    """
    def __init__(self, env, policy, num_local_steps, visualise, predictors, envWrap,
                    noReward, bonus_bound, obs_mean, obs_std, horizon):
        threading.Thread.__init__(self)
        self.queue = queue.Queue(5)  # ideally, should be 1. Mostly doesn't matter in our case.
        self.num_local_steps = num_local_steps
        self.env = env
        self.last_features = None
        self.policy = policy
        self.daemon = True
        self.sess = None
        self.summary_writer = None
        self.visualise = visualise
        self.predictors = predictors
        self.envWrap = envWrap
        self.noReward = noReward
        self.bonus_bound = bonus_bound
        self.obs_mean = obs_mean
        self.obs_std = obs_std
        self.horizon = horizon

    def start_runner(self, sess, summary_writer):
        self.sess = sess
        self.summary_writer = summary_writer
        self.start()

    def run(self):
        with self.sess.as_default():
            self._run()

    def _run(self):
        rollout_provider = env_runner(self.env, self.policy, self.num_local_steps,
                                        self.summary_writer, self.visualise, self.predictors,
                                        self.envWrap, self.noReward, self.bonus_bound, self.obs_mean,
                                        self.obs_std, self.horizon)
        while True:
            # the timeout variable exists because apparently, if one worker dies, the other workers
            # won't die with it, unless the timeout is set to some large number.  This is an empirical
            # observation.

            self.queue.put(next(rollout_provider), timeout=600.0)


def env_runner(env, policy, num_local_steps, summary_writer, render, predictors,
                envWrap, noReward, bonus_bound, obs_mean, obs_std, horizon):
    """
    The logic of the thread runner.  In brief, it constantly keeps on running
    the policy, and as long as the rollout exceeds a certain length, the thread
    runner appends the policy to the queue.
    """
    last_state = env.reset()
    last_features = policy.get_initial_features()  # reset lstm memory
    state_memory = [-1]*(horizon-1) + [last_state] # keep track of past states (max prediction horizon deep) 
    action_memory = [-1]*horizon # keep track of past actions (max prediction horizon deep)
    length = 0
    rewards = 0
    values = 0
    if len(predictors) != 0:
        ep_bonuses = {h:list() for h in range(horizon)}

    while True:
        terminal_end = False
        rollout = PartialRollout(len(predictors) != 0)
        old_rollout = None # keep the old rollout around so you can add the final future prediction rewards
        rollout_hold_steps = 0 # countdown on extra steps to hold the ready-to-yield rollout (to wait for the extra prediction bonuses to come in)
        for local_step in range(num_local_steps):
            
            # run policy
            fetched = policy.act(last_state, *last_features)
            action, value_, features = fetched[0], fetched[1], fetched[2:]

            # run environment: get action_index from sampled one-hot 'action'
            stepAct = action.argmax()
            action_memory.append(action); action_memory.pop(0) # update action memory
            state, reward, terminal, info = env.step(stepAct)

            # normalize observations if needed
            if obs_mean is not None and obs_std is not None:
                state = (state-obs_mean)/obs_std

            if noReward:
                reward = 0.0
            if render:
                env.render()

            curr_tuple = [last_state, action, reward, value_, terminal, last_features] 
            if len(predictors) != 0:
                if local_step != 0:
                    multistep_bonuses.append([None]*horizon) # add a list to collect horizon rewards for this current state in the next iteration
                else:
                    multistep_bonuses = [[None]*horizon] # [ [r_1, r_2 ... r_k], ... ] rollouts corresponding to each (s_t, a_t)

                for h in range(horizon): # go through each predictor
                    if h < local_step+1: # data available to make prediction
                        state_h = state_memory[-(h+1)]
                        action_seq = action_memory[-(h+1):]
                        bonus_h = predictors[h].pred_bonus(state_h, state, np.concatenate(action_seq))
                        multistep_bonuses[-(h+1)][h] = bonus_h
                        ep_bonuses[h].append(bonus_h)

                # if bonus_bound > 0 and bonus > bonus_bound:
                #     bonus = 0.0

                # curr_tuple += [bonus, state, multistep_bonuses]
                curr_tuple += [0, state, multistep_bonuses]

            # collect the experience
            rollout.add(*curr_tuple)
            rewards += reward
            length += 1
            values += value_[0]

            last_state = state
            state_memory.append(last_state); state_memory.pop(0) # update state memory
            last_features = features

            timestep_limit = env.spec.tags.get('wrapper_config.TimeLimit.max_episode_steps')
            if timestep_limit is None: timestep_limit = env.spec.timestep_limit
            if terminal or length >= timestep_limit:
                # prints summary of each life if envWrap==True else each game
                print('-'*100)
                print("EPISODE FINISHED. Sum of shaped rewards: {}. Length: {}".format(rewards, length))
                if len(predictors) != 0:
                    for h in range(horizon):
                        print("\t [{}-step prediction] Total bonus: {}".format(h+1, sum(ep_bonuses[h])))
                    multistep_bonuses = [[None]*horizon] # [ [r_1, r_2 ... r_k], ... ] rollouts corresponding to each (s_t, a_t)
                print('-'*100)
                if 'distance' in info: print('Mario Distance Covered:', info['distance'])
                length = 0
                rewards = 0
                terminal_end = True
                last_features = policy.get_initial_features()  # reset lstm memory
                # TODO: don't reset when gym timestep_limit increases, bootstrap -- doesn't matter for atari?
                # reset only if it hasn't already reseted
                if length >= timestep_limit or not env.metadata.get('semantics.autoreset'):
                    last_state = env.reset()
                    state_memory = [-1]*(horizon-1) + [last_state] # keep track of past states (max prediction horizon deep) 
                    action_memory = [-1]*horizon # keep track of past actions (max prediction horizon deep)

            if info:
                # summarize full game including all lives (even if envWrap=True)
                summary = tf.Summary()
                for k, v in info.items():
                    summary.value.add(tag=k, simple_value=float(v))
                if terminal:
                    summary.value.add(tag='global/episode_value', simple_value=float(values))
                    values = 0
                    if len(predictors) != 0:
                        for h in range(horizon):
                            summary.value.add(tag='global/{}step_episode_bonus'.format(h), simple_value=float(sum(ep_bonuses[h])))
                            histogram = tf.HistogramProto() 
                            histogram.min = float(np.min(ep_bonuses[h])) 
                            histogram.max = float(np.max(ep_bonuses[h]))
                            histogram.num = len(ep_bonuses[h]) 
                            histogram.sum = float(np.sum(ep_bonuses[h])) 
                            counts, edges = np.histogram(ep_bonuses[h], bins = 100)
                            for edge in edges[1:]: 
                                histogram.bucket_limit.append(edge) 
                            for count in counts:
                                histogram.bucket.append(count) 
                            summary.value.add(tag='global/{}step_bonus_hist'.format(h), histo=histogram)

                summary_writer.add_summary(summary, policy.global_step.eval())
                summary_writer.flush()

            if terminal_end:
                ep_bonuses = {h:list() for h in range(horizon)}
                break

        if not terminal_end:
            rollout.r = policy.value(last_state, *last_features)

        # once we have enough experience, yield it, and have the ThreadRunner place it on a queue
        yield rollout


class A3C(object):
    def __init__(self, env, task, visualise, unsupType, envWrap=False, designHead='universe', noReward=False, 
                bonus_bound=None, adv_norm=False, obs_mean=None, obs_std=None, r_std_running=False, backup_bound=None, horizon=1, mstep_mode='sum'):
        """
        An implementation of the A3C algorithm that is reasonably well-tuned for the VNC environments.
        Below, we will have a modest amount of complexity due to the way TensorFlow handles data parallelism.
        But overall, we'll define the model, specify its inputs, and describe how the policy gradients step
        should be computed.
        """
        self.task = task
        self.local_steps = 0
        self.unsup = unsupType is not None
        self.envWrap = envWrap
        self.env = env
        self.adv_norm = adv_norm
        self.r_std_running = r_std_running
        if self.r_std_running:
            self.r_std_running = RunningMeanStd()
        self.backup_bound = backup_bound
        self.vpreds = tf.placeholder(tf.float32, [None], name="vpreds") # placeholder for V predictions passed in through feed dict
        self.horizon = horizon
        self.mstep_mode = mstep_mode

        # broadcast multistep mode
        if self.horizon == 1:
            print('FORWARD MODEL SETUP: 1 step prediction')
        else:
            print('FORWARD MODEL SETUP: {} step prediction, with mode {}'.format(self.horizon, self.mstep_mode))

        # initialize forward/inverse models central server parameters
        predictors = list()
        numaction = env.action_space.n
        worker_device = "/job:worker/task:{}/cpu:0".format(task)

        with tf.device(tf.train.replica_device_setter(1, worker_device=worker_device)):
            with tf.variable_scope("global"):
                self.network = LSTMPolicy(env.observation_space.shape, numaction, designHead)
                self.global_step = tf.get_variable("global_step", [], tf.int32, initializer=tf.constant_initializer(0, dtype=tf.int32),
                                                   trainable=False)
                if self.unsup:
                    self.ap_network = dict()
                    for h in range(self.horizon):
                        with tf.variable_scope("predictor_{}".format(h+1)):
                            if 'state' in unsupType:
                                self.ap_network = StatePredictor(env.observation_space.shape, numaction, designHead, unsupType)
                            else:
                                # self.ap_network = StateActionPredictor(env.observation_space.shape, numaction, designHead, self.horizon)
                                self.ap_network[h] = StateActionPredictor(env.observation_space.shape, numaction, designHead, h+1)

        # initialize forward/inverse models local worker parameters
        with tf.device(worker_device):
            with tf.variable_scope("local"):
                self.local_network = pi = LSTMPolicy(env.observation_space.shape, numaction, designHead)
                pi.global_step = self.global_step
                if self.unsup:
                    self.local_ap_network = dict()
                    for h in range(self.horizon):
                        with tf.variable_scope("predictor_{}".format(h+1)):
                            if 'state' in unsupType:
                                self.local_ap_network = predictor = StatePredictor(env.observation_space.shape, numaction, designHead, unsupType)
                            else:
                                # self.local_ap_network = predictor = StateActionPredictor(env.observation_space.shape, numaction, designHead, self.horizon)
                                self.local_ap_network[h] = predictor = StateActionPredictor(env.observation_space.shape, numaction, designHead, h+1)
                                predictors.append(predictor)

            # Computing a3c loss: https://arxiv.org/abs/1506.02438
            self.ac = tf.placeholder(tf.float32, [None, numaction], name="ac")
            self.adv = tf.placeholder(tf.float32, [None], name="adv")
            self.r = tf.placeholder(tf.float32, [None], name="r")
            log_prob_tf = tf.nn.log_softmax(pi.logits)
            prob_tf = tf.nn.softmax(pi.logits)
            # 1) the "policy gradients" loss:  its derivative is precisely the policy gradient
            # notice that self.ac is a placeholder that is provided externally.
            # adv will contain the advantages, as calculated in process_rollout
            pi_loss = - tf.reduce_mean(tf.reduce_sum(log_prob_tf * self.ac, 1) * self.adv)  # Eq (19)
            pi_loss = tf.Print(pi_loss, [pi_loss], message='LP: ')
            # 2) loss of value function: l2_loss = (x-y)^2/2
            vf_loss = 0.5 * tf.reduce_mean(tf.square(pi.vf - self.r))  # Eq (28)
            vf_loss = tf.Print(vf_loss, [vf_loss], message='LV: ')
            # 3) entropy to ensure randomness
            entropy = - tf.reduce_mean(tf.reduce_sum(prob_tf * log_prob_tf, 1)) * constants['ENTROPY_BETA']
            entropy = tf.Print(entropy, [entropy], message='Entropy: ')
            # final a3c loss: lr of critic is half of actor
            self.loss = pi_loss + 0.5 * vf_loss - entropy

            # compute gradients
            grads = tf.gradients(self.loss * constants['ROLLOUT_MAXLEN'], pi.var_list)  # batchsize=ROLLOUT MAXLENGTH. Factored out to make hyperparams not depend on it.

            # computing predictor loss
            if self.unsup:
                if 'state' in unsupType:
                    self.predloss = constants['PREDICTION_LR_SCALE'] * predictor.forwardloss
                    predgrads = tf.gradients(self.predloss * constants['ROLLOUT_MAXLEN'], predictor.var_list)  # batchsize=ROLLOUT MAXLENGTH. Factored out to make hyperparams not depend on it.
                else:
                    self.predlosses = [0.0]*horizon # prediction losses for each network
                    predgrads = [0.0]*horizon # prediction gradients for each network
                    for h in range(horizon):
                        predictor = predictors[h]
                        self.predlosses[h] = constants['PREDICTION_LR_SCALE'] * (predictor.invloss * (1-constants['FORWARD_LOSS_WT']) +
                                                                    predictor.forwardloss * constants['FORWARD_LOSS_WT'])
                        predgrads[h] = tf.gradients(self.predlosses[h] * constants['ROLLOUT_MAXLEN'], predictor.var_list)  # batchsize=ROLLOUT MAXLENGTH. Factored out to make hyperparams not depend on it.

                # do not backprop to policy
                if constants['POLICY_NO_BACKPROP_STEPS'] > 0:
                    grads = [tf.scalar_mul(tf.to_float(tf.greater(self.global_step, constants['POLICY_NO_BACKPROP_STEPS'])), grads_i)
                                    for grads_i in grads]


            self.runner = RunnerThread(env, pi, constants['ROLLOUT_MAXLEN'], visualise,
                                        predictors, envWrap, noReward, bonus_bound, obs_mean, obs_std, horizon)

            # storing summaries
            bs = tf.to_float(tf.shape(pi.x)[0])
            if use_tf12_api:
                tf.summary.scalar("model/policy_loss", pi_loss)
                tf.summary.scalar("model/value_loss", vf_loss)
                tf.summary.scalar("model/entropy", entropy)
                tf.summary.image("model/state", pi.x)  # max_outputs=10
                tf.summary.scalar("model/grad_global_norm", tf.global_norm(grads))
                tf.summary.scalar("model/var_global_norm", tf.global_norm(pi.var_list))
                tf.summary.scalar("model/vpreds", tf.reduce_mean(self.vpreds))
                tf.summary.scalar("model/advantages", tf.reduce_mean(self.adv))
                if self.unsup:
                    for h in range(horizon):
                        tf.summary.scalar("model/predloss_{}".format(h+1), self.predlosses[h])
                    if 'action' in unsupType:
                        for h in range(horizon):
                            predictor = predictors[h]
                            tf.summary.scalar("model/inv_loss_{}".format(h+1), predictor.invloss)
                            tf.summary.scalar("model/forward_loss_{}".format(h+1), predictor.forwardloss)
                            tf.summary.scalar("model/predgrad_global_norm_{}".format(h+1), tf.global_norm(predgrads[h]))
                            tf.summary.scalar("model/predvar_global_norm_{}".format(h+1), tf.global_norm(predictor.var_list))
                # if self.task == 0 and self.local_steps % 25800 == 0:
                #     tf.summary.histogram('global/vpreds', self.vpreds)
                #     tf.summary.histogram('global/advantages', self.adv)
                self.summary_op = tf.summary.merge_all()
            else:
                tf.scalar_summary("model/policy_loss", pi_loss)
                tf.scalar_summary("model/value_loss", vf_loss)
                tf.scalar_summary("model/entropy", entropy)
                tf.image_summary("model/state", pi.x)
                tf.scalar_summary("model/grad_global_norm", tf.global_norm(grads))
                tf.scalar_summary("model/var_global_norm", tf.global_norm(pi.var_list))
                tf.scalar_summary("model/vpreds", tf.reduce_mean(self.vpreds))
                tf.scalar_summary("model/advantages", tf.reduce_mean(self.adv))
                if self.unsup:
                    for h in range(horizon):
                        tf.scalar_summary("model/predloss_{}".format(h+1), self.predlosses[h])
                    if 'action' in unsupType:
                        for h in range(horizon):
                            predictor = predictors[h]
                            tf.scalar_summary("model/inv_loss_{}".format(h+1), predictor.invloss)
                            tf.scalar_summary("model/forward_loss_{}".format(h+1), predictor.forwardloss)
                            tf.scalar_summary("model/predgrad_global_norm_{}".format(h+1), tf.global_norm(predgrads[h]))
                            tf.scalar_summary("model/predvar_global_norm_{}".format(h+1), tf.global_norm(predictor.var_list))
                # if self.task == 0 and self.local_steps % 25800 == 0:
                #     tf.histogram_summary('global/vpreds', self.vpreds)
                #     tf.histogram_summary('global/advantages', self.adv)
                self.summary_op = tf.merge_all_summaries()

            # clip gradients
            grads, _ = tf.clip_by_global_norm(grads, constants['GRAD_NORM_CLIP'])
            grads_and_vars = list(zip(grads, self.network.var_list))
            if self.unsup:
                for h in range(horizon):
                    predgrads[h], _ = tf.clip_by_global_norm(predgrads[h], constants['GRAD_NORM_CLIP'])
                    pred_grads_and_vars = list(zip(predgrads[h], self.ap_network[h].var_list))
                    grads_and_vars = grads_and_vars + pred_grads_and_vars

            # update global step by batch size
            inc_step = self.global_step.assign_add(tf.shape(pi.x)[0])

            # each worker has a different set of adam optimizer parameters
            # TODO: make optimizer global shared, if needed
            print("Optimizer: ADAM with lr: %f" % (constants['LEARNING_RATE']))
            print("Input observation shape: ",env.observation_space.shape)
            opt = tf.train.AdamOptimizer(constants['LEARNING_RATE'])
            self.train_op = tf.group(opt.apply_gradients(grads_and_vars), inc_step)

            # copy weights from the parameter server to the local model
            sync_var_list = [v1.assign(v2) for v1, v2 in zip(pi.var_list, self.network.var_list)]
            if self.unsup:
                for h in range(horizon):
                    predictor = predictors[h]
                    sync_var_list += [v1.assign(v2) for v1, v2 in zip(predictor.var_list, self.ap_network[h].var_list)]
            self.sync = tf.group(*sync_var_list)

            # initialize extras
            self.summary_writer = None

    def start(self, sess, summary_writer):
        self.runner.start_runner(sess, summary_writer)
        self.summary_writer = summary_writer

    def pull_batch_from_queue(self):
        """
        Take a rollout from the queue of the thread runner.
        """
        # get top rollout from queue (FIFO)
        rollout = self.runner.queue.get(timeout=600.0)
        while not rollout.terminal:
            try:
                # Now, get remaining *available* rollouts from queue and append them into
                # the same one above. If queue.Queue(5): len=5 and everything is
                # superfast (not usually the case), then all 5 will be returned and
                # exception is raised. In such a case, effective batch_size would become
                # constants['ROLLOUT_MAXLEN'] * queue_maxlen(5). But it is almost never the
                # case, i.e., collecting  a rollout of length=ROLLOUT_MAXLEN takes more time
                # than get(). So, there are no more available rollouts in queue usually and
                # exception gets always raised. Hence, one should keep queue_maxlen = 1 ideally.
                # Also note that the next rollout generation gets invoked automatically because
                # its a thread which is always running using 'yield' at end of generation process.
                # To conclude, effective batch_size = constants['ROLLOUT_MAXLEN']
                rollout.extend(self.runner.queue.get_nowait())
            except queue.Empty:
                break
        return rollout

    def process(self, sess):
        """
        Process grabs a rollout that's been produced by the thread runner,
        and updates the parameters.  The update is then sent to the parameter
        server.
        """
        sess.run(self.sync)  # copy weights from shared to local
        rollout = self.pull_batch_from_queue()
        batch = process_rollout(rollout, gamma=constants['GAMMA'], lambda_=constants['LAMBDA'], clip=self.envWrap, adv_norm=self.adv_norm, 
            r_std_running=self.r_std_running, backup_bound=self.backup_bound, horizon=self.horizon, mstep_mode=self.mstep_mode)
        should_compute_summary = self.task == 0 and self.local_steps % 11 == 0
        self.r_std_running = batch.r_std_running

        if should_compute_summary:
            fetches = [self.summary_op, self.train_op, self.global_step]
        else:
            fetches = [self.train_op, self.global_step]

        feed_dict = {
            self.local_network.x: batch.si,
            self.ac: batch.a[0],
            self.adv: batch.adv,
            self.r: batch.r,
            self.local_network.state_in[0]: batch.features[0],
            self.local_network.state_in[1]: batch.features[1],
            self.vpreds: batch.vpreds
        }
        if self.unsup:
            feed_dict[self.local_network.x] = batch.si[:-1]
            for h in range(self.horizon):
                feed_dict[self.local_ap_network[h].s1] = batch.si[:len(rollout.states)-h]
                feed_dict[self.local_ap_network[h].s2] = batch.si[h+1:]
                feed_dict[self.local_ap_network[h].asample] = batch.a[h]

        fetched = sess.run(fetches, feed_dict=feed_dict)
        if batch.terminal:
            print("Global Step Counter: %d"%fetched[-1])

        if should_compute_summary:
            self.summary_writer.add_summary(tf.Summary.FromString(fetched[0]), fetched[-1])
            self.summary_writer.flush()
        self.local_steps += 1
