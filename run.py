
import os, sys
from copy import deepcopy
import json, ast
import argparse
from six.moves import shlex_quote
from params_manager import params_object
from utilities.text_utils import *

# ------------------------------------------- DEFAULTS ------------------------------------------- #
# paths
LOG_PATH="./experiment_log.xlsx"
RESULTS_PATH_DOOM="./curiosity/results/icm/doom"
RESULTS_PATH_MARIO="./curiosity/results/icm/mario"


# default arguments for training operations
TRAINING_PARAMS = {
    'num_workers': 20,
    'remotes':None,
    'env_id':'doom',
    'log_dir':'tmp/model',
    'dry_run':False,
    'mode':'tmux',
    'visualise':True,
    'envWrap':True,
    'designHead':'universe',
    'unsup':None,
    'noReward':False,
    'noLifeReward':True,
    'expName':'a3c',
    'expId':0,
    'savio':False,
    'default':False,
    'pretrain':None,
    'record_frequency':300, # in episodes
    'record_dir':'tmp/model/videos',
    'bonus_bound':-1.0,
    'adv_norm':False,
    'obs_norm':False,
    'rew_norm':False,
    'backup_bound':-1.0
}

# arguments with 'action = store_true' in train.py
STORE_TRUE_TRAIN = {'dry_run', 'envWrap', 'noReward', 'noLifeReward', 
                    'savio', 'default', 'adv_norm', 'obs_norm', 'rew_norm'}

# default arguments for demonstration operations
DEMO_PARAMS = {
    'ckpt':None, # defined during execution
    'outdir':None, # defined during execution
    'env_id':'doom',
    'record':True,
    'render':True,
    'num_episodes':2,
    'greedy':False,
    'random':False,
    'obs_norm':False,
    'demo':False
}

# arguments with 'action = store_true' in demo.py
STORE_TRUE_DEMO = {'record', 'render', 'greedy', 'random', 'obs_norm', 'demo'}

# default arguments for inference operation
INF_PARAMS = {
    'log_dir':None, # defined during execution
    'out_dir':None, # defined during execution
    'env_id':'doom',
    'record':True,
    'recordSignal':False,
    'render':True,
    'envWrap':True,
    'designHead':'universe',
    'num_episodes':2,
    'noop':False,
    'acRepeat':0,
    'greedy':False,
    'random':False,
    'default':True,
    'demo':False
}

# arguments with 'action = store_true' in inference.py
STORE_TRUE_INF = {'record', 'render', 'recordSignal', 'envWrap', 'noop', 'greedy', 'random', 'default', 'demo'}

# default arguments for plot operations
PLOT_PARAMS = {
    'plot_tags':None,
    'output_dir':None,
    'x_axis':'step',
    'y_axis':'global/episode_reward',
    'ave_runs':True,
    'ave_tags':False,
    'x_increment':10000,
    'left_x':0,
    'right_x':10000000,
    'span':150.0
}

# arguments with 'action = store_true' in plot.py
STORE_TRUE_PLOT = {}


# ------------------------------------------- ARGUMENTS ------------------------------------------- #

parser = argparse.ArgumentParser(description='Run high level tasks')

# GENERAL ARGUMENTS
parser.add_argument('-op', default=None, help='Which operation to run: swap, train, demo, or plot')
parser.add_argument('-registry', default='experiment_log.xlsx', help='Path to excel file containing information for all experiments')
parser.add_argument('-tag', default=None, help='The name associated with the model you want to swap, run or do inference on')

# TRAINING OP ARGUMENTS
parser.add_argument('-num-workers', type=int, default=20, help='Number of workers')
parser.add_argument('-remotes', default=None, help='The address of pre-existing VNC servers and rewarders to use (e.g. -r vnc://localhost:5900+15900,vnc://localhost:5901+15901)')
parser.add_argument('-env-id', type=str, default='doom', help='Environment id')
parser.add_argument('-log-dir', type=str, default='tmp/model', help='Log directory path')
parser.add_argument('-dry-run', type=bool, default=False, help='Print out commands rather than executing them')
parser.add_argument('-mode', type=str, default='tmux', help='tmux: run workers in a tmux session. nohup: run workers with nohup. child: run workers as child processes')
parser.add_argument('-visualise', type=bool, default=True, help='Visualise the gym environment by running env.render() between each timestep')
parser.add_argument('-envWrap', type=bool, default=True, help='Preprocess input in env_wrapper (no change in input size or network)')
parser.add_argument('-designHead', type=str, default='universe', help='Network design head: nips or nature or doom or universe (default)')
parser.add_argument('-unsup', type=str, default=None, help='Unsup. exploration mode: action or state or stateAenc or None')
parser.add_argument('-noReward', type=bool, default=False, help='Remove all extrinsic reward')
parser.add_argument('-noLifeReward', type=bool, default=True, help='Remove all negative reward (in doom: it is living reward)')
parser.add_argument('-expName', type=str, default='a3c', help='Experiment tmux session-name. Default a3c')
parser.add_argument('-expId', type=int, default=0, help='Experiment Id >=0. Needed while runnig more than one run per machine')
parser.add_argument('-savio', type=bool, default=False, help='Savio or KNL cpu cluster hacks')
parser.add_argument('-default', type=bool, default=False, help='run with default params')
parser.add_argument('-pretrain', type=str, default=None, help='Checkpoint dir (generally ..../train/) to load from')
parser.add_argument('-record-frequency', type=int, default=300, help='Interval (in episode ids) between saved videos. 300 works well for mario, 50 is better for doom.')
parser.add_argument('-record-dir', type=str, default='tmp/model/videos', help='Path to directory where training videos should be saved')
parser.add_argument('-bonus-bound', type=float, default=-1.0, help='Intrinsic reward bound. If reward is above this, its set to 0')
parser.add_argument('-adv-norm', type=bool, default=False, help='Normalize batch advantages after each rollout')
parser.add_argument('-obs-norm', type=bool, default=False, help='Locally tandardize observations (pixelwise, individually by channel)')
parser.add_argument('-rew-norm', type=bool, default=False, help='Normalize batch rewards by dividing by running standard deviation')
parser.add_argument('-backup-bound', type=float, default=-1.0, help="Bound the intrinsic reward discounted sum (backup term) before computing network targets")

# DEMO OP ARGUMENTS
# parser.add_argument('--ckpt', default='../models/doom/doom_ICM', help='Checkpoint name')
# parser.add_argument('--outdir', default='../models/output', help='Output log directory')
# parser.add_argument('--env-id', default='doom', help='Environment id')
# parser.add_argument('--record', type=bool, default=True, help='Record the policy running video')
# parser.add_argument('--render', type=bool, default=False, help='Render the gym environment video online')
# parser.add_argument('--num-episodes', type=int, default=2, help='Number of episodes to run')
# parser.add_argument('--greedy', type=bool, default=False, help='Default sampled policy. This option does argmax')
# parser.add_argument('--random', type=bool, default=False, help='Default sampled policy. This option does random policy')
# parser.add_argument('--obs-norm', type=bool, default=False, help='Whether or not you should normalize the observations')
# parser.add_argument('--demo', type=bool, default=False, help='Whether or not youre using the demo model provided by the authors')

# INFERENCE OP ARGUMENTS
parser.add_argument('--log-dir', default="tmp/doom", help='input model directory')
parser.add_argument('--out-dir', default=None, help='output log directory. Default: log_dir/inference/')
parser.add_argument('--env-id', default="PongDeterministic-v3", help='Environment id')
parser.add_argument('--record', type=bool, default=True, help="Record the gym environment video -- user friendly")
parser.add_argument('--recordSignal', type=bool, default=False, help="Record images of true processed input to network")
parser.add_argument('--render', type=bool, default=True, help="Render the gym environment video online")
parser.add_argument('--envWrap', type=bool, default=True, help="Preprocess input in env_wrapper (no change in input size or network)")
parser.add_argument('--designHead', type=str, default='universe', help="Network deign head: nips or nature or doom or universe(default)")
parser.add_argument('--num-episodes', type=int, default=2, help="Number of episodes to run")
parser.add_argument('--noop', type=bool, default=False, help="Add 30-noop for inference too (recommended by Nature paper, don't know?)")
parser.add_argument('--acRepeat', type=int, default=0, help="Actions to be repeated at inference. 0 means default. applies iff envWrap is True.")
parser.add_argument('--greedy', type=bool, default=False, help="Default sampled policy. This option does argmax.")
parser.add_argument('--random', type=bool, default=False, help="Default sampled policy. This option does random policy.")
parser.add_argument('--default', type=bool, default=True, help="run with default params")
parser.add_argument('--demo', type=bool, default=False, help="Whether or not youre using the demo model provided by the authors")

# PLOT OP ARGUMENTS
parser.add_argument('--plot-tags', default=None, help='Usertags you want to plot (separated by + signs)')
parser.add_argument('--output-dir', default=None, help='Usertag directory where you want to save plots')
parser.add_argument('--x-axis', default='step', help='The x axis value')
parser.add_argument('--y-axis', default='global/episode_reward', help='The scalar value being plotted')
parser.add_argument('--ave-runs', default=True, help='Whether or not each plot tag should plot train_0, or all runs averaged')
parser.add_argument('--ave-tags', default=False, help='Whether or not specified plot tags should be plotted individually, or averaged')
parser.add_argument('--x-increment', type=int, default=10000, help='Spacing between x-axis values. Only used when averaging runs, or averaging plot tags')
parser.add_argument('--left-x', type=int, default=0, help='Leftmost x-axis value (timestep usually).')
parser.add_argument('--right-x', type=int, default=10000000, help='Rightmost x-axis value (timestep usually).')
parser.add_argument('--span', type=float, default=150.0, help='Smoothing parameter for EWMA. 150 for Doom, 500 for Mario is safe.')


# ------------------------------------------- DATABASE ------------------------------------------- #

# Training Parameters
class ParameterObject(params_object.ParamsObject):
    ''' Abstract parent class used to store items in the database '''
    @property
    def projectName(self):
        return 'params'

    @property
    def ignoreHashKeys(self):
        ignore = list()
        return ignore
    
class TrainingParams(ParameterObject):
    ''' Object used to store specific training parameters '''
    def __init__(self, uParams=dict(), usertag=None):
        super(TrainingParams, self).__init__(params=uParams, usertag=usertag)

    def default_params(self):
        return deepcopy(TRAINING_PARAMS)

class ExperimentParams(ParameterObject):
    ''' Object used to store an experiment, and the db ID of associated parameters '''
    def __init__(self, training_params, usertag=None):
        self._training_params = training_params
        super(ExperimentParams, self).__init__(params=dict(), usertag=usertag)

    def default_params(self):
        dParams = dict()
        dParams['params_hash'] = self._training_params.hash_name() 
        return dParams

    def get_paths(self):
        paths = dict()
        paths['experiment_hash'] = self.hash_name()
        return paths 


# ------------------------------------------- MAIN METHOD ------------------------------------------- #

def generate_commands(args):
    ''' Generate relevant commands based on the specified operation '''
    py_cmd = 'python'
    py3_cmd = 'python3'
    commands = list() # commands that will eventually be executed
    params = dict() # training or demo parameters fed to demo.py or train.py
    usertag = str() # unique identifier corresponding to an experiment
    save = False # whether or not to ask about saving parameters and updating experiment log

    if args.op == None: 
        print('---- No operation specified: use the -op flag')

    elif args.op not in {'train', 'swap', 'demo', 'inference', 'plot', 'wr'}:
        print('---- Operation not available (mispelled?)')
        
    # RUN TRAINING OP
    elif args.op == 'train':
        
        if os.path.isdir('./curiosity/src/tmp'):
            if sys.version_info[0] == 2: user_inp = raw_input('MODEL FILES PRESENT, RESUME TRAINING? -> (Y/N): ')
            elif sys.version_info[0] == 3: user_inp = input('MODEL FILES PRESENT, RESUME TRAINING? -> (Y/N): ')
            if user_inp != 'Y': 
                user_inp = raw_input('REMOVE TMP FOLDER? -> (Y/N): ')
                if user_inp != 'Y':
                    print('---- Exiting with no changes')
                elif user_inp == 'Y':
                    os.system('rm -r ./curiosity/src/tmp')
                    print('---- Removed. Exiting.')
            elif user_inp == 'Y':
                if os.path.exists('./curiosity/src/tmp/usertag.txt'):
                    with open('./curiosity/src/tmp/usertag.txt', 'r') as usertag_file: 
                        usertag = usertag_file.read().strip()
                    params_hash = get_value(usertag, 'params_id', args.registry)
                    train = TrainingParams()
                    params = train.find_by_id(params_hash.encode('utf-8')).next()
                    # create training command (directory changes before this is run)
                    commands.append('{} train.py {}'.format(py_cmd, dict_to_command(params, STORE_TRUE_TRAIN, TRAINING_PARAMS, args.op)))
                    print('---- Restarting existing training session: {}'.format(usertag))
                else:
                    print('---- No usertag.txt file in current tmp directory')

        elif args.tag: # new model with old model parameters specified

            # check if the requested old model exists in the log
            sourcetag = args.tag
            if get_value(sourcetag, 'usertag', args.registry) is None:
                print('---- Model {} does not exist in experiment log'.format(sourcetag))
                return commands, params, sourcetag, save

            # create a new usertag for this seed and retrieve parameters from database
            usertag = create_usertag(args)
            params_hash = get_value(sourcetag, 'params_id', args.registry)
            train = TrainingParams()
            params_doc = train.find_by_id(params_hash.encode('utf-8')).next()
            params = {k.encode('ascii'): unicode(v).encode('ascii') for k,v in params_doc.iteritems() if k in TRAINING_PARAMS}
            if params:
                save = True
                commands.append('{} train.py {}'.format(py_cmd, dict_to_command(params, STORE_TRUE_TRAIN, TRAINING_PARAMS, args.op)))
                print('---- Starting a new training session {} with same parameters as {}'.format(usertag, sourcetag))
            else:
                print('---- Couldnt find appropriate hash in experiment log')

        else: # new model with new parameters
            save = True
            usertag = create_usertag(args)

            # override arguments
            params = deepcopy(TRAINING_PARAMS)
            arg_set = set(vars(args))
            for arg in vars(args):
                if arg in TRAINING_PARAMS: params[arg] = getattr(args, arg)

            # create training command
            cmd = '{} train.py {}'.format(py_cmd, dict_to_command(params, STORE_TRUE_TRAIN, TRAINING_PARAMS, args.op)) # directory changes before this is run
            commands.append(cmd)
            print('---- Starting a new training session with new parameters: {}'.format(usertag))


    # RUN DEMO OP
    elif args.op == 'demo':

        if not args.tag: # no usertag specified
            print('---- No model tag specified')
            return commands, params, usertag, save

        if not os.path.isdir('{}/{}'.format(result_path, args.tag)):
            print('---- Model {} not found'.format(args.tag))
            return commands, params, usertag, save

        # set paths
        result_path = RESULTS_PATH_DOOM
        demo_path = '../results/icm/doom'
        if 'mario' in args.tag.lower():
            result_path = RESULTS_PATH_MARIO
            demo_path = '../results/icm/mario'

        # find most recent meta file
        nums = list()
        for file in os.listdir('{}/{}/model/train'.format(result_path, args.tag)):
            nums += [int(n.split('-')[-1]) for n in file.split('.') if numeric(n.split('-')[-1])]
        nums.sort()
        highest_global_step = int(nums[-1])

        # override default arguments
        params = deepcopy(DEMO_PARAMS)
        arg_set = set(vars(args))
        for arg in vars(args):
            if arg == 'outdir':
                params[arg] = '{}/{}/demos'.format(demo_path, args.tag)
            elif arg == 'ckpt':
                if args.demo:
                    params[arg] = '../results/mario/mario_ICM'
                else:
                    params[arg] = '{}/{}/model/train/model.ckpt-{}'.format(demo_path, args.tag, highest_global_step)
            elif arg in params: params[arg] = getattr(args, arg)

        # generate demo command
        cmd = '{} demo.py {}'.format(py_cmd, dict_to_command(params, STORE_TRUE_DEMO, DEMO_PARAMS, args.op)) 
        commands.append(cmd)
        print('---- Starting demo with model {} on env {}'.format(args.tag, params['env_id']))

    # RUN INFERENCE OP
    elif args.op == 'inference':

        if not args.tag: # no usertag specified
            print('---- No model tag specified')
            return commands, params, usertag, save

        # set paths
        result_path = RESULTS_PATH_DOOM
        inf_path = '../results/icm/doom'
        if 'mario' in args.tag.lower():
            result_path = RESULTS_PATH_MARIO
            inf_path = '../results/icm/mario'

        if not os.path.isdir('{}/{}'.format(result_path, args.tag)):
            print('---- Model {} not found'.format(args.tag))
            return commands, params, usertag, save

        # find most recent meta file
        nums = list()
        for file in os.listdir('{}/{}/model/train'.format(result_path, args.tag)):
            nums += [int(n.split('-')[-1]) for n in file.split('.') if numeric(n.split('-')[-1])]
        nums.sort()
        highest_global_step = int(nums[-1])

        # override default arguments
        params = deepcopy(INF_PARAMS)
        arg_set = set(vars(args))
        for arg in vars(args):
            if arg == 'log_dir':
                params[arg] = '{}/{}/model'.format(inf_path, args.tag)
            elif arg == 'out_dir':
                params[arg] = '{}/{}/demos'.format(inf_path, args.tag)
            elif arg in params: params[arg] = getattr(args, arg)

        # generate infernece command
        cmd = '{} inference.py {}'.format(py_cmd, dict_to_command(params, STORE_TRUE_INF, INF_PARAMS, args.op))
        commands.append(cmd)
        print('---- Starting demo with model {} on env {}'.format(args.tag, params['env_id']))

    # SWAP TRAINING FILES
    elif args.op == 'swap':

        # set paths
        result_path = RESULTS_PATH_DOOM # results path
        src_path = './curiosity/src' # source path

        # tmp is present, ask to store it
        if os.path.isdir('{}/tmp'.format(src_path)):
            if sys.version_info[0] == 2: inp = raw_input('MODEL FILES PRESENT, STORE THEM? -> (Y/N): ')
            elif sys.version_info[0] == 3: inp = input('MODEL FILES PRESENT, STORE THEM? -> (Y/N): ')
            if inp == 'N': 
                print('---- Exiting with no changes')
            else:
                with open('{}/tmp/usertag.txt'.format(src_path)) as file: usertag = file.read()
                if 'mario' in usertag.lower():
                    result_path = './curiosity/results/icm/mario'
                print('---- Storing tmp directory in {}/{}'.format(result_path, usertag))
                commands.append('mv {}/tmp {}/{}'.format(src_path, result_path, usertag))
        # no usertag provided
        elif not args.tag:
            print('---- No tmp directory detected, and no model tag specified')
        # no tmp folder, target usertag provided
        else:
            if not os.path.isdir('{}/{}'.format(result_path, args.tag)):
                print('---- Model {} not found'.format(args.tag))
                return commands, params, usertag, save
            commands.append('mv {}/{} {}/'.format(result_path, args.tag, src_path + '/tmp'))


    # PLOTTING OP
    elif args.op == 'plot':

        if args.plot_tags is None:
            print('---- No model tags specified')
            return commands, params, usertag, save
        params = deepcopy(PLOT_PARAMS)
        arg_set = set(vars(args))
        params['output_dir'] = getattr(args, 'output_dir')
        for arg in vars(args):
            if arg == 'output_dir' and getattr(args, arg) is None:
                params[arg] = args.plot_tags.split('+')[0]
            elif arg in PLOT_PARAMS: params[arg] = getattr(args, arg)
        cmd = '{} plot.py {}'.format(py3_cmd, dict_to_command(params, STORE_TRUE_PLOT, PLOT_PARAMS, args.op))
        commands.append(cmd)

    return commands, params, usertag, save


def run():
    args = parser.parse_args()

    commands, params, usertag, save_params = generate_commands(args)
    commands = '\n'.join(commands)

    if len(commands) == 0: return

    print('#'*70)
    print('Generated commands:')
    print('-'*70)
    print(commands)
    print('#'*70)
    if sys.version_info[0] == 2: confirmation = raw_input('RUN COMMANDS? -> (Y/N): ')
    elif sys.version_info[0] == 3: confirmation = input('RUN COMMANDS? -> (Y/N): ')
    if confirmation == 'Y':
        if args.op not in  {'swap', 'plot'}: os.chdir('./curiosity/src')
        os.system(commands)
        if args.op not in  {'swap', 'plot'}: os.chdir('../../')

        # ask if you should save params/register the experiment
        if save_params and os.path.isdir('./curiosity/src/tmp'):

            if sys.version_info[0] == 2: save = raw_input('SAVE/REGISTER EXPERIMENT? -> (Y/N): ')
            elif sys.version_info[0] == 3: save = input('SAVE/REGISTER EXPERIMENT -> (Y/N): ')
            if save != 'Y':
                print('---- Exiting without updating registry or database')
                return

            os.system('sleep 5') # wait to be sure the experiment is successfully launched
            if params and usertag:
                repeat_count = get_count(usertag, args.registry)
                if repeat_count == 0:
                    trainParams = TrainingParams(params)
                    expParams = ExperimentParams(trainParams, usertag=usertag)
                    exp_id = expParams.get_paths()['experiment_hash']
                    params_id = expParams.default_params()['params_hash']
                else: 
                    repeat_count = int(repeat_count)
                    exp_id = get_value(usertag, 'experiment_id', args.registry)
                    params_id = get_value(usertag, 'params_id', args.registry)
                update_registry(args, usertag, repeat_count, exp_id, params_id)
                print('---- Successfully stored parameters, and updated registry')

    else:
        print('---- Exiting with no changes')

if __name__ == '__main__':
    run()





