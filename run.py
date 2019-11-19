
import os, sys
from copy import deepcopy
from datetime import datetime
import pytz
import argparse
from six.moves import shlex_quote
from params_manager import params_object


# ------------------------------------------- DEFAULTS ------------------------------------------- #

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
    'bonus_bound':-1.0
}

# arguments with 'action = store_true' in train.py
STORE_TRUE_TRAIN = {'dry_run', 'envWrap', 'noReward', 'noLifeReward', 'savio', 'default'}

# default arguments for demonstration operations
DEMO_PARAMS = {
    'ckpt':None, # defined during execution
    'outdir':None, # defined during execution
    'env_id':'doom',
    'record':True,
    'render':True,
    'num_episodes':2,
    'greedy':False,
    'random':False
}

# arguments with 'action = store_true' in demo.py
STORE_TRUE_DEMO = {'record', 'render', 'greedy', 'random'}

# line indexes for the registry file
REGISTRY = { 
    'none_dense':0, 
    'none_sparse':1,
    'none_verySparse':2,
    'icm_dense':3,
    'icm_sparse':4, 
    'icm_verySparse':5,
    'icmpix_dense':6,
    'icmpix_sparse':7,
    'icmpix_verySparse':8
}


# ------------------------------------------- ARGUMENTS ------------------------------------------- #

parser = argparse.ArgumentParser(description='Run high level tasks')

# GENERAL ARGUMENTS
parser.add_argument('-op', default=None, help='Which operation to run: swap, train, or demo')
parser.add_argument('-registry', default='experiment_log.txt', help='Path to .txt file containing usertags and hash keys for all experiments')

# TRAINING OP ARGUMENTS
parser.add_argument('-num-workers', type=int, default=20, help="Number of workers")
parser.add_argument('-remotes', default=None, help='The address of pre-existing VNC servers and rewarders to use (e.g. -r vnc://localhost:5900+15900,vnc://localhost:5901+15901)')
parser.add_argument('-env-id', type=str, default="doom", help="Environment id")
parser.add_argument('-log-dir', type=str, default="tmp/model", help="Log directory path")
parser.add_argument('-dry-run', type=bool, default=False, help="Print out commands rather than executing them")
parser.add_argument('-mode', type=str, default='tmux', help="tmux: run workers in a tmux session. nohup: run workers with nohup. child: run workers as child processes")
parser.add_argument('-visualise', type=bool, default=True, help="Visualise the gym environment by running env.render() between each timestep")
parser.add_argument('-envWrap', type=bool, default=True, help="Preprocess input in env_wrapper (no change in input size or network)")
parser.add_argument('-designHead', type=str, default='universe', help="Network deign head: nips or nature or doom or universe(default)")
parser.add_argument('-unsup', type=str, default=None, help="Unsup. exploration mode: action or state or stateAenc or None")
parser.add_argument('-noReward', type=bool, default=False, help="Remove all extrinsic reward")
parser.add_argument('-noLifeReward', type=bool, default=True, help="Remove all negative reward (in doom: it is living reward)")
parser.add_argument('-expName', type=str, default='a3c', help="Experiment tmux session-name. Default a3c")
parser.add_argument('-expId', type=int, default=0, help="Experiment Id >=0. Needed while runnig more than one run per machine")
parser.add_argument('-savio', type=bool, default=False, help="Savio or KNL cpu cluster hacks")
parser.add_argument('-default', type=bool, default=False, help="run with default params")
parser.add_argument('-pretrain', type=str, default=None, help="Checkpoint dir (generally ..../train/) to load from")
parser.add_argument('-record-frequency', type=int, default=300, help="Interval (in episodes) between saved videos")
parser.add_argument('-record-dir', type=str, default='tmp/model/videos', help="Path to directory where training videos should be saved")
parser.add_argument('-bonus-bound', type=float, default=-1.0, help="Intrinsic reward bound. If reward is above this, it's set to 0")

# DEMO OP ARGUMENTS
parser.add_argument('--ckpt', default="../models/doom/doom_ICM", help='checkpoint name')
parser.add_argument('--outdir', default="../models/output", help='Output log directory')
parser.add_argument('--env-id', default="doom", help='Environment id')
parser.add_argument('--record', type=bool, default=True, help="Record the policy running video")
parser.add_argument('--render', type=bool, default=False, help="Render the gym environment video online")
parser.add_argument('--num-episodes', type=int, default=2, help="Number of episodes to run")
parser.add_argument('--greedy', type=bool, default=False, help="Default sampled policy. This option does argmax.")
parser.add_argument('--random', type=bool, default=False, help="Default sampled policy. This option does random policy.")

# SWAP OP ARGUMENTS
parser.add_argument('-tag', default=None, help='The name associated with the model')


# ------------------------------------------- DATABASE ------------------------------------------- #

# Training Parameters
class ParameterObject(params_object.ParamsObject):
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
    ''' Object used to store an experiment, and a link to associated parameters '''
    def __init__(self, training_params, usertag=None):
        self._training_params = training_params
        super(ExperimentParams, self).__init__(params=dict(), usertag=usertag)

    def default_params(self):
        dParams = dict()
        dParams['params_hash'] = self._training_params.hash_name()
        return dParams

    def get_paths(self):
        paths = dict()
        paths['exp'] = self.hash_name()
        return paths 


# ------------------------------------------- UTILITIES ------------------------------------------- #

flatten = lambda l: [item for sublist in l for item in sublist]

def create_usertag(args):
    ''' Create a usertag of the following format: alg_setting_# (e.g. icm_dense_3) '''
    with open(args.registry, 'r') as registry:
        registry_text = registry.readlines()

        # Algorithm choice (none, icm, icmpix)
        if args.unsup == None: algo = 'none'
        elif args.unsup == 'action': algo = 'icm'
        elif 'state' in args.unsup.lower(): algo = 'icmpix'

        # Reward setting (dense, sparse, verySparse)
        if 'very' in args.env_id.lower(): setting='verySparse'
        elif 'sparse' in args.env_id.lower(): setting='sparse'
        else: setting='dense'

        combo = '{}_{}'.format(algo, setting) 
        trial_number = str(int(registry_text[REGISTRY[combo]].strip()[-1])+1) 
        usertag = '{}_{}'.format(combo, trial_number)
        return usertag

def dict_to_command(args, mode):
    ''' Convert a dictionary into a sequence of command line arguments '''
    if mode == 'train': 
        store_true = STORE_TRUE_TRAIN
        params = TRAINING_PARAMS
    elif mode == 'demo': 
        store_true = STORE_TRUE_DEMO
        params = DEMO_PARAMS

    cmd = ''
    for argument in args:
        value = args[argument]
        if argument not in params:
            continue
        if argument in store_true:
            if value == True: cmd += '--{} '.format(argument.replace('_', '-'))
            continue
        cmd += '--{} {} '.format(argument.replace('_', '-'), value)
    return cmd

def update_experiment_count(filename, usertag):
    ''' Update the experiment counter in the registry file '''
    usertag_pieces = usertag.split('_')
    trial_number = int(usertag_pieces.pop(-1)) # increment the count by one
    alg_and_setting = '_'.join(usertag_pieces)

    file = open(filename, 'r')
    lines = file.readlines()
    file.close()
    for i, line in enumerate(lines):
        if line.split('=')[0].strip(' \n') == alg_and_setting:
            lines[i] = alg_and_setting + ' = ' + str(trial_number) + '\n'
    file = open(filename, 'w')
    file.write(''.join(lines))
    file.close()

def get_model_info(args):
    with open('./curiosity/src/tmp/usertag.txt', 'r') as usertag_file: usertag = usertag_file.read()
    with open(args.registry, 'r') as registry: lines = registry.readlines()
    for i, line in enumerate(lines):
        stripped_line = flatten([n.strip().split(':') for n in line.split('|')])
        if usertag in stripped_line:
            params_hash = stripped_line[stripped_line.index('training_params_id') + 1].strip()
    return usertag, params_hash

def store_and_register(args, params, usertag):
    ''' Store parameters in database, and update the experiment registry '''
    # store parameters in database
    trainParams = TrainingParams(params)
    expParams = ExperimentParams(trainParams, usertag=usertag)
    storage_path = expParams.get_paths()

    # register the experiment in the registry
    update_experiment_count(args.registry, usertag)
    with open(args.registry, 'a')as registry:
        time = datetime.now(pytz.timezone('US/Eastern'))
        new_entry = '{} | {} | experiment_id: {} | training_params_id: {} \n'.format(time, \
            usertag, storage_path['exp'], expParams.default_params()['params_hash'])
        registry.write(new_entry)

    # create a usertag.txt file in tmp with the correct usertag and datetime
    with open('./curiosity/src/tmp/usertag.txt', 'w+') as file:
        file.write('{}'.format(usertag))


# ------------------------------------------- MAIN METHOD ------------------------------------------- #

def generate_commands(args):
    ''' Generate relevant commands based on the specified operation '''
    py_cmd = 'python'
    commands = list()
    params = dict()
    usertag = str()

    if args.op == None: 
        print('---- No operation specified: use the -op flag')

    elif args.op not in {'train', 'swap', 'demo'}:
        print('---- Operation does not match any available (mispelled?)')
        
    # RUN TRAINING OP
    elif args.op == 'train':
        
        if os.path.isdir('./curiosity/src/tmp'):
            inp = raw_input('tmp directory present -> resume training? (Y/N): ')
            if inp == 'N': 
                print('---- Exiting with no changes')
            elif inp == 'Y':

                usertag, params_hash = get_model_info(args)

                # grab parameters from db
                train = TrainingParams()
                doc = train.find_by_id(params_hash)
                params = dict_to_command(doc.next(), args.op)
                params = params.replace('_', '-') # remember to replace '_' with '-'

                # create training command (directory changes before this is run)
                commands.append('{} train.py {}'.format(py_cmd, params))
                print('---- Restarting existing training session')

        elif args.tag: # no tmp file, old model parameters specified
            # NEW TRAINING OP USING EXISTING PARAMS
            print('n')

        else: # if no tmp file, start a new training op

            usertag = create_usertag(args)

            # override arguments
            params = deepcopy(TRAINING_PARAMS)
            arg_set = set(vars(args))
            for arg in vars(args):
                if arg in TRAINING_PARAMS: params[arg] = getattr(args, arg)

            # create training command
            cmd = '{} train.py {}'.format(py_cmd, dict_to_command(params, args.op)) # directory changes before this is run
            commands.append(cmd)


    # RUN DEMO OP
    elif args.op == 'demo':

        demo_path = '../results/icm'
        result_path = './curiosity/results/icm'

        if not args.tag: # no usertag specified
            print('---- No model tag specified')
            return list(), dict(), str()

        if not os.path.isdir('{}/{}'.format(result_path, args.tag)):
            print('---- Model {} not found'.format(args.tag))
            return list(), dict(), str()

        # find most recent meta file
        nums = list()
        def numeric(chars):
            try:
                float(chars)
                return True
            except ValueError:
                return False
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
                params[arg] = '{}/{}/model/train/model.ckpt-{}'.format(demo_path, args.tag, highest_global_step)
            elif arg in params: params[arg] = getattr(args, arg)

        # generate demo command
        cmd = '{} demo.py {}'.format(py_cmd, dict_to_command(params, args.op))
        commands.append(cmd)
        print('---- Starting demo with model {} on env {}'.format(args.tag, params['env_id']))


    # SWAP TRAINING FILES
    elif args.op == 'swap':

        src_path = './curiosity/src' # source path
        result_path = './curiosity/results/icm' # results path
        if os.path.isdir('{}/tmp'.format(src_path)): # tmp is present, ask to store it
            inp = raw_input('Tmp directory detected, store it? (Y/N): ')
            if inp == 'N': 
                print('---- Exiting with no changes')
            else:
                with open('{}/tmp/usertag.txt'.format(src_path)) as file: usertag = file.read()
                print('---- Storing tmp directory in {}/{}'.format(result_path, usertag))
                commands.append('mv {}/tmp {}/{}'.format(src_path, result_path, usertag))
        elif not args.tag: # no tmp folder, no usertag
            print('---- No tmp directory detected, and no model tag specified')
        else: # no tmp folder, target usertag provided
            if not os.path.isdir('{}/{}'.format(result_path, args.tag)):
                print('Model {} not found'.format(args.tag))
                return list(), dict(), str()
            commands.append('mv {}/{} {}/'.format(result_path, args.tag, src_path + '/tmp'))

    return commands, params, usertag


def run():
    args = parser.parse_args()

    commands, params, usertag = generate_commands(args)
    commands = '\n'.join(commands)

    if len(commands) == 0: return

    print('-'*50)
    print('Generated commands:')
    print('~'*50)
    print(commands)
    print('-'*50)
    confirmation = raw_input('RUN COMMANDS? (Y/N): ')

    if confirmation == 'Y':
        if args.op != 'swap': os.chdir('./curiosity/src')
        os.system(commands)
        if args.op != 'swap': os.chdir('../../')

        # ask if you should save params/register the experiment
        if args.op == 'train' and os.path.isdir('./curiosity/src/tmp'):
            save = raw_input('SAVE/REGISTER EXPERIMENT? (Y/N): ') 
            if save != 'Y':
                print('---- Exiting without updating registry or database')
                return

            os.system('sleep 5') # wait to be sure the experiment is successfully launched
            if params and usertag:
                store_and_register(args, params, usertag)
                print('---- Successfully stored parameters, and updated registry')

    else:
        print('---- Exiting with no changes')

if __name__ == '__main__':
    run()



