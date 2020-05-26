from __future__ import print_function
import numpy as np
import tensorflow as tf
import tensorflow.contrib.rnn as rnn
from constants import constants

class LSTMCellWrapper(rnn.rnn_cell.RNNCell):
  def __init__(self, inner_cell):
     super(LSTMCellWrapper, self).__init__()
     self._inner_cell = inner_cell
  
  @property
  def state_size(self):
     return self._inner_cell.state_size
  
  @property
  def output_size(self):
    return (self._inner_cell.state_size, self._inner_cell.output_size)

  def __call__(self, input, *args, **kwargs):
    output, next_state = self._inner_cell(input, *args, **kwargs)
    emit_output = (next_state, output)
    return emit_output, next_state

def normalized_columns_initializer(std=1.0):
    def _initializer(shape, dtype=None, partition_info=None):
        out = np.random.randn(*shape).astype(np.float32)
        out *= std / np.sqrt(np.square(out).sum(axis=0, keepdims=True))
        return tf.constant(out)
    return _initializer


def cosineLoss(A, B, name):
    ''' A, B : (BatchSize, d) '''
    dotprod = tf.reduce_sum(tf.multiply(tf.nn.l2_normalize(A,1), tf.nn.l2_normalize(B,1)), 1)
    loss = 1-tf.reduce_mean(dotprod, name=name)
    return loss


def flatten(x, horizon=None):
    if horizon is not None: # lstm flattening: [batch, horizon, 3, 3, 32] ==> [batch, horizon, 288]
        return tf.reshape(x, [-1, horizon, np.prod(x.get_shape().as_list()[2:])])
    else: # standard flattening: [batch, 3, 3, 32] ==> [batch, 288]
        return tf.reshape(x, [-1, np.prod(x.get_shape().as_list()[1:])])


def conv2d(x, num_filters, name, filter_size=(3, 3), stride=(1, 1), pad="SAME", dtype=tf.float32, collections=None):
    with tf.variable_scope(name):
        stride_shape = [1, stride[0], stride[1], 1]
        filter_shape = [filter_size[0], filter_size[1], int(x.get_shape()[3]), num_filters]

        # there are "num input feature maps * filter height * filter width"
        # inputs to each hidden unit
        fan_in = np.prod(filter_shape[:3])
        # each unit in the lower layer receives a gradient from:
        # "num output feature maps * filter height * filter width" /
        #   pooling size
        fan_out = np.prod(filter_shape[:2]) * num_filters
        # initialize weights with random weights
        w_bound = np.sqrt(6. / (fan_in + fan_out))

        w = tf.get_variable("W", filter_shape, dtype, tf.random_uniform_initializer(-w_bound, w_bound),
                            collections=collections)
        b = tf.get_variable("b", [1, 1, 1, num_filters], initializer=tf.constant_initializer(0.0),
                            collections=collections)
        return tf.nn.conv2d(x, w, stride_shape, pad) + b


def deconv2d(x, out_shape, name, filter_size=(3, 3), stride=(1, 1), pad="SAME", dtype=tf.float32, collections=None, prevNumFeat=None):
    with tf.variable_scope(name):
        num_filters = out_shape[-1]
        prevNumFeat = int(x.get_shape()[3]) if prevNumFeat is None else prevNumFeat
        stride_shape = [1, stride[0], stride[1], 1]
        # transpose_filter : [height, width, out_channels, in_channels]
        filter_shape = [filter_size[0], filter_size[1], num_filters, prevNumFeat]

        # there are "num input feature maps * filter height * filter width"
        # inputs to each hidden unit
        fan_in = np.prod(filter_shape[:2]) * prevNumFeat
        # each unit in the lower layer receives a gradient from:
        # "num output feature maps * filter height * filter width"
        fan_out = np.prod(filter_shape[:3])
        # initialize weights with random weights
        w_bound = np.sqrt(6. / (fan_in + fan_out))

        w = tf.get_variable("W", filter_shape, dtype, tf.random_uniform_initializer(-w_bound, w_bound),
                            collections=collections)
        b = tf.get_variable("b", [num_filters], initializer=tf.constant_initializer(0.0),
                            collections=collections)
        deconv2d = tf.nn.conv2d_transpose(x, w, tf.pack(out_shape), stride_shape, pad)
        # deconv2d = tf.reshape(tf.nn.bias_add(deconv2d, b), deconv2d.get_shape())
        return deconv2d

def conv3d(x, num_filters, name, filter_size=(3, 3), stride=(1, 1), pad="SAME", dtype=tf.float32, collections=None, horizon=1):
    with tf.variable_scope(name):
        stride_shape = [1, 1, stride[0], stride[1], 1] # [batch, depth, height, width, channel]
        filter_shape = [horizon, filter_size[0], filter_size[1], int(x.get_shape()[4]), num_filters] # [depth, height, width, in_channels, out_channels]

        # there are "depth * filter height * filter width * num input channels"
        # inputs to each hidden unit
        fan_in = np.prod(filter_shape[:4])
        # each unit in the lower layer receives a gradient from:
        # "depth * filter height * filter width * num output features" /
        #   pooling size
        fan_out = np.prod(filter_shape[:3]) * num_filters
        # initialize weights with random weights
        w_bound = np.sqrt(6. / (fan_in + fan_out))

        w = tf.get_variable("W", filter_shape, dtype, tf.random_uniform_initializer(-w_bound, w_bound),
                            collections=collections)
        b = tf.get_variable("b", [1, 1, 1, 1, num_filters], initializer=tf.constant_initializer(0.0),
                            collections=collections)
        return tf.nn.conv3d(x, w, stride_shape, pad) + b


def linear(x, size, name, initializer=None, bias_init=0):
    w = tf.get_variable(name + "/w", [x.get_shape()[1], size], initializer=initializer)
    b = tf.get_variable(name + "/b", [size], initializer=tf.constant_initializer(bias_init))
    return tf.matmul(x, w) + b


def categorical_sample(logits, d):
    value = tf.squeeze(tf.multinomial(logits - tf.reduce_max(logits, [1], keep_dims=True), 1), [1])
    return tf.one_hot(value, d)


def inverseUniverseHead(x, final_shape, nConvs=4):
    ''' universe agent example
        input: [None, 288]; output: [None, 42, 42, 1];
    '''
    print('Using inverse-universe head design')
    bs = tf.shape(x)[0]
    deconv_shape1 = [final_shape[1]]
    deconv_shape2 = [final_shape[2]]
    for i in range(nConvs):
        deconv_shape1.append((deconv_shape1[-1]-1)/2 + 1)
        deconv_shape2.append((deconv_shape2[-1]-1)/2 + 1)
    inshapeprod = np.prod(x.get_shape().as_list()[1:]) / 32.0
    assert(inshapeprod == deconv_shape1[-1]*deconv_shape2[-1])
    # print('deconv_shape1: ',deconv_shape1)
    # print('deconv_shape2: ',deconv_shape2)

    x = tf.reshape(x, [-1, deconv_shape1[-1], deconv_shape2[-1], 32])
    deconv_shape1 = deconv_shape1[:-1]
    deconv_shape2 = deconv_shape2[:-1]
    for i in range(nConvs-1):
        x = tf.nn.elu(deconv2d(x, [bs, deconv_shape1[-1], deconv_shape2[-1], 32],
                        "dl{}".format(i + 1), [3, 3], [2, 2], prevNumFeat=32))
        deconv_shape1 = deconv_shape1[:-1]
        deconv_shape2 = deconv_shape2[:-1]
    x = deconv2d(x, [bs] + final_shape[1:], "dl4", [3, 3], [2, 2], prevNumFeat=32)
    return x


def universeHead(x, nConvs=4):
    ''' universe agent example (2d convolution)
        input: [None, 42, 42, 1]; output if 2d: [None, 288];
    '''
    print('Using universe head design (2D conv)')
    for i in range(nConvs):
        x = tf.nn.elu(conv2d(x, 32, "2d_l{}".format(i + 1), [3, 3], [2, 2]))
        # print('Loop{} '.format(i+1),tf.shape(x))
        # print('Loop{}'.format(i+1),x.get_shape())
    x = flatten(x)
    return x

def universeHead3d(x, nConvs=4, horizon=1):
    ''' universe agent example (3d convolution)
        input: [None, horizon, 42, 42, 1]; output if 3d: [None, horizon, 288]
    '''
    print('Using universe head design (3D conv)')
    for i in range(nConvs):
        x = tf.nn.elu(conv3d(x, 32, "3d_l{}".format(i + 1), [3, 3], [2, 2], horizon=horizon))
        # print('Loop{} '.format(i+1),tf.shape(x))
        # print('Loop{}'.format(i+1),x.get_shape())
    x = flatten(x, horizon=horizon)
    return x

def nipsHead(x, lstm=False):
    ''' DQN NIPS 2013 and A3C paper
        input: [None, 84, 84, 4]; output: [None, 2592] -> [None, 256];
    '''
    print('Using nips head design')
    x = tf.nn.relu(conv2d(x, 16, "l1", [8, 8], [4, 4], pad="VALID"))
    x = tf.nn.relu(conv2d(x, 32, "l2", [4, 4], [2, 2], pad="VALID"))
    x = flatten(x)
    x = tf.nn.relu(linear(x, 256, "fc", normalized_columns_initializer(0.01)))
    return x


def natureHead(x, lstm=False):
    ''' DQN Nature 2015 paper
        input: [None, 84, 84, 4]; output: [None, 3136] -> [None, 512];
    '''
    print('Using nature head design')
    x = tf.nn.relu(conv2d(x, 32, "l1", [8, 8], [4, 4], pad="VALID"))
    x = tf.nn.relu(conv2d(x, 64, "l2", [4, 4], [2, 2], pad="VALID"))
    x = tf.nn.relu(conv2d(x, 64, "l3", [3, 3], [1, 1], pad="VALID"))
    x = flatten(x)
    x = tf.nn.relu(linear(x, 512, "fc", normalized_columns_initializer(0.01)))
    return x


def doomHead(x, lstm=False):
    ''' Learning by Prediction ICLR 2017 paper
        (their final output was 64 changed to 256 here)
        input: [None, 120, 160, 1]; output: [None, 1280] -> [None, 256];
    '''
    print('Using doom head design')
    x = tf.nn.elu(conv2d(x, 8, "l1", [5, 5], [4, 4]))
    x = tf.nn.elu(conv2d(x, 16, "l2", [3, 3], [2, 2]))
    x = tf.nn.elu(conv2d(x, 32, "l3", [3, 3], [2, 2]))
    x = tf.nn.elu(conv2d(x, 64, "l4", [3, 3], [2, 2]))
    x = flatten(x)
    x = tf.nn.elu(linear(x, 256, "fc", normalized_columns_initializer(0.01)))
    return x


class LSTMPolicy(object):
    def __init__(self, ob_space, ac_space, designHead='universe'):
        self.x = x = tf.placeholder(tf.float32, [None] + list(ob_space), name='x')
        size = 256
        if designHead == 'nips':
            x = nipsHead(x)
        elif designHead == 'nature':
            x = natureHead(x)
        elif designHead == 'doom':
            x = doomHead(x)
        elif 'tile' in designHead:
            x = universeHead(x, nConvs=2)
        else:
            x = universeHead(x)

        # introduce a "fake" batch dimension of 1 to do LSTM over time dim
        x = tf.expand_dims(x, [0])
        lstm = rnn.rnn_cell.BasicLSTMCell(size, state_is_tuple=True)
        self.state_size = lstm.state_size
        step_size = tf.shape(self.x)[:1]

        c_init = np.zeros((1, lstm.state_size.c), np.float32)
        h_init = np.zeros((1, lstm.state_size.h), np.float32)
        self.state_init = [c_init, h_init]

        c_in = tf.placeholder(tf.float32, [1, lstm.state_size.c], name='c_in')
        h_in = tf.placeholder(tf.float32, [1, lstm.state_size.h], name='h_in')
        self.state_in = [c_in, h_in]

        state_in = rnn.rnn_cell.LSTMStateTuple(c_in, h_in)
        lstm_outputs, lstm_state = tf.nn.dynamic_rnn(
            lstm, x, initial_state=state_in, sequence_length=step_size,
            time_major=False)
        lstm_c, lstm_h = lstm_state

        x = tf.reshape(lstm_outputs, [-1, size])
        self.vf = tf.reshape(linear(x, 1, "value", normalized_columns_initializer(1.0)), [-1])
        self.state_out = [lstm_c[:1, :], lstm_h[:1, :]]

        # [0, :] means pick action of first state from batch. Hardcoded b/c
        # batch=1 during rollout collection. Its not used during batch training.
        self.logits = linear(x, ac_space, "action", normalized_columns_initializer(0.01))
        self.sample = categorical_sample(self.logits, ac_space)[0, :]
        self.probs = tf.nn.softmax(self.logits, dim=-1)[0, :]

        self.var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, tf.get_variable_scope().name)

        # Demo related collections (used for inference once model is trained)
        tf.add_to_collection('probs', self.probs)
        tf.add_to_collection('sample', self.sample)
        tf.add_to_collection('state_out_0', self.state_out[0])
        tf.add_to_collection('state_out_1', self.state_out[1])
        tf.add_to_collection('vf', self.vf)

    def get_initial_features(self):
        # Call this function to get reseted lstm memory cells
        return self.state_init

    def act(self, ob, c, h):
        sess = tf.get_default_session()
        return sess.run([self.sample, self.vf] + self.state_out,
                        {self.x: [ob], self.state_in[0]: c, self.state_in[1]: h})

    def act_inference(self, ob, c, h):
        sess = tf.get_default_session()
        return sess.run([self.probs, self.sample, self.vf] + self.state_out,
                        {self.x: [ob], self.state_in[0]: c, self.state_in[1]: h})

    def value(self, ob, c, h):
        sess = tf.get_default_session()
        return sess.run(self.vf, {self.x: [ob], self.state_in[0]: c, self.state_in[1]: h})[0]


class StateActionPredictor(object):
    def __init__(self, ob_space, ac_space, designHead='universe', horizon=1):
        # input: s1,s2: : [None, h, w, ch] (usually ch=1 or 4)
        # asample: 1-hot encoding of sampled action from policy: [None, ac_space]
        input_shape = [None] + list(ob_space)
        self.s1 = phi1 = tf.placeholder(tf.float32, input_shape)
        self.s2 = phi2 = tf.placeholder(tf.float32, input_shape)
        self.asample = asample = tf.placeholder(tf.float32, [None, ac_space*horizon])

        # feature encoding: phi1, phi2: [None, LEN]
        size = 256
        if designHead == 'nips':
            phi1 = nipsHead(phi1)
            with tf.variable_scope(tf.get_variable_scope(), reuse=True):
                phi2 = nipsHead(phi2)
        elif designHead == 'nature':
            phi1 = natureHead(phi1)
            with tf.variable_scope(tf.get_variable_scope(), reuse=True):
                phi2 = natureHead(phi2)
        elif designHead == 'doom':
            phi1 = doomHead(phi1)
            with tf.variable_scope(tf.get_variable_scope(), reuse=True):
                phi2 = doomHead(phi2)
        elif 'tile' in designHead:
            phi1 = universeHead(phi1, nConvs=2)
            with tf.variable_scope(tf.get_variable_scope(), reuse=True):
                phi2 = universeHead(phi2, nConvs=2)
        else:
            phi1 = universeHead(phi1)
            with tf.variable_scope(tf.get_variable_scope(), reuse=True):
                phi2 = universeHead(phi2)

        # inverse model: g(phi1,phi2) -> a_inv: [None, ac_space]
        g = tf.concat(1,[phi1, phi2])
        g = tf.nn.relu(linear(g, size, "g1", normalized_columns_initializer(0.01)))
        aindex = tf.argmax(asample, axis=1)  # aindex: [batch_size,]
        logits = linear(g, ac_space, "glast", normalized_columns_initializer(0.01))
        self.invloss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
                                        logits, aindex), name="invloss")
        self.ainvprobs = tf.nn.softmax(logits, dim=-1)

        # forward model: f(phi1,asample) -> phi2
        # Note: no backprop to asample of policy: it is treated as fixed for predictor training
        f = tf.concat(1, [phi1, asample])
        f = tf.nn.relu(linear(f, size, "f1", normalized_columns_initializer(0.01)))
        f = linear(f, phi1.get_shape()[1].value, "flast", normalized_columns_initializer(0.01))
        self.forwardloss = 0.5 * tf.reduce_mean(tf.square(tf.subtract(f, phi2)), name='forwardloss')
        # self.forwardloss = 0.5 * tf.reduce_mean(tf.sqrt(tf.abs(tf.subtract(f, phi2))), name='forwardloss')
        # self.forwardloss = cosineLoss(f, phi2, name='forwardloss')
        self.forwardloss = self.forwardloss * 288.0  # lenFeatures=288. Factored out to make hyperparams not depend on it.

        # variable list
        self.var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, tf.get_variable_scope().name)

    def pred_bonus(self, s1, s2, asample):
        '''
        returns bonus predicted by forward model
            input: s1,s2: [h, w, ch], asample: [ac_space] 1-hot encoding
            output: scalar bonus
        '''
        sess = tf.get_default_session()
        # error = sess.run([self.forwardloss, self.invloss],
        #     {self.s1: [s1], self.s2: [s2], self.asample: [asample]})
        # print('ErrorF: ', error[0], ' ErrorI:', error[1])
        error = sess.run(self.forwardloss,
            {self.s1: [s1], self.s2: [s2], self.asample: [asample]})
        error = error * constants['PREDICTION_BETA']
        return error

class LSTMPredictor(object):
    def __init__(self, ob_space, ac_space, designHead='universe', horizon=1):
        # input: s1,s2: : [None, h, w, ch] (usually ch=1 or 4)
        # asample: 1-hot encoding of sampled action from policy: [None, ac_space]
        input_shape = [None] + list(ob_space)

        # state inputs for forard or inverse model
        self.s1 = phi1 = tf.placeholder(tf.float32, input_shape, name='s1')
        self.s2 = phi2 = tf.placeholder(tf.float32, input_shape, name='s2')

        # action input for forward model
        self.asample = asample = tf.placeholder(tf.float32, [None, ac_space])

        # feature encoding: phi1, phi2: [None, LEN]
        size = 256
        if designHead == 'nips':
            phi1 = nipsHead(phi1)
            with tf.variable_scope(tf.get_variable_scope(), reuse=True):
                phi2 = nipsHead(phi2)
        elif designHead == 'nature':
            phi1 = natureHead(phi1)
            with tf.variable_scope(tf.get_variable_scope(), reuse=True):
                phi2 = natureHead(phi2)
        elif designHead == 'doom':
            phi1 = doomHead(phi1)
            with tf.variable_scope(tf.get_variable_scope(), reuse=True):
                phi2 = doomHead(phi2)
        elif 'tile' in designHead:
            phi1 = universeHead(phi1, nConvs=2)
            with tf.variable_scope(tf.get_variable_scope(), reuse=True):
                phi2 = universeHead(phi2, nConvs=2)
        else:
            phi1 = universeHead(phi1)
            with tf.variable_scope(tf.get_variable_scope(), reuse=True):
                phi2 = universeHead(phi2)

        # INVERSE MODEL: g(phi1,phi2) -> a_inv: [None, ac_space]
        g = tf.concat(1, [phi1, phi2])
        g = tf.nn.relu(linear(g, size, "g1", normalized_columns_initializer(0.01)))

        aindex = tf.argmax(asample, axis=1)  # aindex: [batch_size,]
        logits = linear(g, ac_space, "glast", normalized_columns_initializer(0.01))
        self.invloss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
                                        logits, aindex), name="invloss")
        self.ainvprobs = tf.nn.softmax(logits, dim=-1)[0, :]

        # FORWARD MODEL: (phi1 + asample) -> [LSTM] -> phi2
        # Note: no backprop to asample of policy: it is treated as fixed for predictor training

        # [batch, 288] concat [batch, 4] ==> [batch, 288 + 4]
        self.batch_size = batch_size = tf.placeholder(tf.int32, name='batch_size')
        x = tf.concat(1, [phi1, asample]) # [batch, 288 + 4]
        x = tf.expand_dims(x, [0]) # [1, batch, 288 + 4]

        # initialize individual lstm cells
        lstm_cell = LSTMCellWrapper(rnn.rnn_cell.LSTMCell(constants['LSTM_PREDICTOR_NUM_UNITS'], state_is_tuple=True))
        # lstm_cell = rnn.rnn_cell.LSTMCell(constants['LSTM_PREDICTOR_NUM_UNITS'], state_is_tuple=True)
        self.state_size = lstm_cell.state_size

        # lstm initial state placeholders
        c_in = tf.placeholder(tf.float32, [1, lstm_cell.state_size.c], name='c_in_lstm')
        h_in = tf.placeholder(tf.float32, [1, lstm_cell.state_size.h], name='h_in_lstm')
        self.state_in = [c_in, h_in]

        c_init = np.zeros((1, lstm_cell.state_size.c), np.float32)
        h_init = np.zeros((1, lstm_cell.state_size.h), np.float32)
        self.state_init = [c_init, h_init]

        # unroll lstm
        state_in = rnn.rnn_cell.LSTMStateTuple(c_in, h_in)
        lstm_outputs, lstm_state = tf.nn.dynamic_rnn(lstm_cell, x, initial_state=state_in, sequence_length=batch_size, time_major=False)
        lstm_all_c = lstm_outputs[0].c # all cell states
        lstm_all_h = lstm_outputs[0].h # all hidden states
        lstm_outputs = lstm_outputs[1]
        lstm_final_c, lstm_final_h = lstm_state # this is the last state out

        lstm_outputs = tf.reshape(lstm_outputs, [-1, 288]) # [batch, 1, 288] ==> [batch, 288]
        self.state_out = [lstm_all_c[0], lstm_all_h[0]]

        # compute loss 
        self.forwardloss = 0.5 * tf.reduce_mean(tf.square(tf.subtract(lstm_outputs, phi2)), axis=1)
        self.forwardloss = tf.reshape(self.forwardloss, [-1]) # [[b1, b2, b3]] ==> [b1, b2, b3]
        self.forwardloss = self.forwardloss * 288.0 # lenFeatures=288. Factored out to make hyperparams not depend on it.

        # variable list
        self.var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, tf.get_variable_scope().name)

    def get_initial_features(self):
        # Call this function to get initial lstm memory cells
        return self.state_init

    def pred_bonus(self, s1, s2, asample, c, h):
        '''
        returns bonus predicted by forward model
            input: s1,s2 shapes: (horizon, h, w, ch), asample shape: (horizon, ac_space) where ac_space is 1-hot encoding
            output: (horizon, scalar bonuses)
        '''
        sess = tf.get_default_session()
        errors_and_features = sess.run([self.forwardloss] + self.state_out, 
                                        {self.s1: s1, 
                                        self.s2: s2, 
                                        self.asample: asample, 
                                        self.state_in[0]: c, 
                                        self.state_in[1]: h, 
                                        self.batch_size: len(s1)})
        errors, c, h = errors_and_features
        errors = errors * constants['PREDICTION_BETA']
        return errors, [c, h]

class StatePredictor(object):
    '''
    Loss is normalized across spatial dimension (42x42), but not across batches.
    It is unlike ICM where no normalization is there across 288 spatial dimension
    and neither across batches.
    '''

    def __init__(self, ob_space, ac_space, designHead='universe', unsupType='state', horizon=1):
        # input: s1,s2: : [None, h, w, ch] (usually ch=1 or 4)
        # asample: 1-hot encoding of sampled action from policy: [None, ac_space]
        input_shape = [None] + list(ob_space)
        self.s1 = phi1 = tf.placeholder(tf.float32, input_shape)
        self.s2 = phi2 = tf.placeholder(tf.float32, input_shape)
        self.asample = asample = tf.placeholder(tf.float32, [None, ac_space*horizon])
        self.stateAenc = unsupType == 'stateAenc'

        # feature encoding: phi1: [None, LEN]
        if designHead == 'universe':
            phi1 = universeHead(phi1)
            if self.stateAenc:
                with tf.variable_scope(tf.get_variable_scope(), reuse=True):
                    phi2_aenc = universeHead(phi2)
        elif 'tile' in designHead:  # for mario tiles
            phi1 = universeHead(phi1, nConvs=2)
            if self.stateAenc:
                with tf.variable_scope(tf.get_variable_scope(), reuse=True):
                    phi2_aenc = universeHead(phi2)
        else:
            print('Only universe designHead implemented for state prediction baseline.')
            exit(1)

        # forward model: f(phi1,asample) -> phi2
        # Note: no backprop to asample of policy: it is treated as fixed for predictor training
        f = tf.concat(1, [phi1, asample])
        f = tf.nn.relu(linear(f, phi1.get_shape()[1].value, "f1", normalized_columns_initializer(0.01)))
        if 'tile' in designHead:
            f = inverseUniverseHead(f, input_shape, nConvs=2)
        else:
            f = inverseUniverseHead(f, input_shape)
        self.forwardloss = 0.5 * tf.reduce_mean(tf.square(tf.subtract(f, phi2)), name='forwardloss')
        if self.stateAenc:
            self.aencBonus = 0.5 * tf.reduce_mean(tf.square(tf.subtract(phi1, phi2_aenc)), name='aencBonus')
        self.predstate = phi1

        # variable list
        self.var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, tf.get_variable_scope().name)

    def pred_state(self, s1, asample):
        '''
        returns state predicted by forward model
            input: s1: [h, w, ch], asample: [ac_space] 1-hot encoding
            output: s2: [h, w, ch]
        '''
        sess = tf.get_default_session()
        return sess.run(self.predstate, {self.s1: [s1],
                                            self.asample: [asample]})[0, :]

    def pred_bonus(self, s1, s2, asample):
        '''
        returns bonus predicted by forward model
            input: s1,s2: [h, w, ch], asample: [ac_space] 1-hot encoding
            output: scalar bonus
        '''
        sess = tf.get_default_session()
        bonus = self.aencBonus if self.stateAenc else self.forwardloss
        error = sess.run(bonus,
            {self.s1: [s1], self.s2: [s2], self.asample: [asample]})
        # print('ErrorF: ', error)
        error = error * constants['PREDICTION_BETA']
        return error
