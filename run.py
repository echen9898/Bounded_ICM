
import os, sys
from datetime import datetime
import pytz
import argparse
from six.moves import shlex_quote
from exp_params_mng import params_object

parser = argparse.ArgumentParser(description='Run high level tasks')

# GENERAL ARGUMENTS
parser.add_argument('-dry-run-highlvl', action='store_true',
                    help='Generate commands and print them, but dont execute anything')
parser.add_argument('-op', default=None,
                    help='Which operation to run: swap, train, or demo')
parser.add_argument('-registry', default='experiment_log.txt',
                    help='Path to .txt file containing usertags and hash keys for all experiments')

general_args = {'dry_run_highlvl', 'op', 'registry'}

# SWAP/DEMO OP ARGUMENTS
parser.add_argument('-tag', default=None, help='The name associated with the model')

swap_args = {'tag'}

# TRAINING OP ARGUMENTS
parser.add_argument('-num-workers', default=20, type=int,
                    help="Number of workers")
parser.add_argument('-remotes', default=None,
                    help='The address of pre-existing VNC servers and '
                         'rewarders to use (e.g. -r vnc://localhost:5900+15900,vnc://localhost:5901+15901).')
parser.add_argument('-env-id', type=str, default="doom",
                    help="Environment id")
parser.add_argument('-log-dir', type=str, default="tmp/model",
                    help="Log directory path")
parser.add_argument('-dry-run', action='store_true',
                    help="Print out commands rather than executing them")
parser.add_argument('-mode', type=str, default='tmux',
                    help="tmux: run workers in a tmux session. nohup: run workers with nohup. child: run workers as child processes")
parser.add_argument('-visualise', action='store_true',
                    help="Visualise the gym environment by running env.render() between each timestep")
parser.add_argument('-envWrap', action='store_true',
                    help="Preprocess input in env_wrapper (no change in input size or network)")
parser.add_argument('-designHead', type=str, default='universe',
                    help="Network deign head: nips or nature or doom or universe(default)")
parser.add_argument('-unsup', type=str, default=None,
                    help="Unsup. exploration mode: action or state or stateAenc or None")
parser.add_argument('-noReward', action='store_true', help="Remove all extrinsic reward")
parser.add_argument('-noLifeReward', action='store_true',
                    help="Remove all negative reward (in doom: it is living reward)")
parser.add_argument('-expName', type=str, default='a3c',
                    help="Experiment tmux session-name. Default a3c.")
parser.add_argument('-expId', type=int, default=0,
                    help="Experiment Id >=0. Needed while runnig more than one run per machine.")
parser.add_argument('-savio', action='store_true',
                    help="Savio or KNL cpu cluster hacks")
parser.add_argument('-default', action='store_true', help="run with default params")
parser.add_argument('-pretrain', type=str, default=None, help="Checkpoint dir (generally ..../train/) to load from.")

training_args = {'num_workers', 'remotes', 'env_id', 'log_dir', 'dry_run', 'mode', 'visualise', 'envWrap',
    'designHead', 'unsup', 'noReward', 'noLifeReward', 'expName', 'expId', 'savio', 'default', 'pretrain'}

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
    ''' Object used to store training parameters.
    '''
    def __init__(self, uParams=dict(), usertag=None):
        super(TrainingParams, self).__init__(params=uParams, usertag=usertag)

    def default_params(self):
        dParams = dict()
        dParams['num_workers'] = 20
        dParams['remotes'] = None
        dParams['env_id'] = 'doom'
        dParams['log_dir'] = 'tmp/model'
        dParams['dry_run'] = False
        dParams['mode'] = 'tmux'
        dParams['visualise'] = False
        dParams['envWrap'] = True
        dParams['designHead'] = 'universe'
        dParams['unsup'] = 'action'
        dParams['noReward'] = False
        dParams['noLifeReward'] = True
        dParams['expName'] = 'a3c'
        dParams['expId'] = 0
        dParams['savio'] = True
        dParams['default'] = False
        dParams['pretrain'] = None
        return dParams

class ExperimentParams(ParameterObject):
    ''' Object used to store a link to a specific training parameter object, 
    for a specific experiment. Can create multiple experiments that run the same
    set of parameters.
    '''
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

def dict_to_command(args):
    ''' Convert a dictionary to a python argument format.
    '''
    cmd = ''
    for k in args: cmd += '--{} {} '.format(k.replace('_', '-'), args[k])
    return cmd

def replace_in_file(filename, key, new_value):
    f = open(filename, "r")
    lines = f.readlines()
    f.close()
    for i, line in enumerate(lines):
        if line.split('=')[0].strip(' \n') == key:
            lines[i] = key + ' = ' + new_value + '\n'
    f = open(filename, "w")
    f.write("".join(lines))
    f.close()

def generate_commands(args):
    ''' Generate relevant commands based on the specified operation.
    '''
    py_cmd = 'python'
    commands = list()

    with open('experiment_log.txt', 'a+') as registry:
        tz = pytz.timezone('US/Eastern') # timezone object
        log_data = registry.readlines() # registry file data
        lines = { # registry file line numbers
            'icm_dense':0, 
            'icm_sparse':1, 
            'icm_verySparse':2
        }

        if args.op == None: 
            print('No operation specified: use the -op flag')
            return []

            
        elif args.op == 'swap': # Swap training files

            src_path = './curiosity/src' # source path
            result_path = './curiosity/results/icm' # results path
            if os.path.isdir('{}/tmp'.format(src_path)): # tmp is present, ask to store it
                inp = raw_input('Tmp directory detected, store it? (Y/N): ')
                if inp == 'N':
                    print('Exiting with no changes')
                    return []
                else:
                    with open('{}/tmp/usertag.txt'.format(src_path)) as file: usertag = file.read()
                    print('Storing tmp directory in {}/{}'.format(result_path, usertag))
                    commands.append('mv {}/tmp {}/{}'.format(src_path, result_path, usertag))
            elif not args.tag: # no tmp folder, no usertag
                print('No tmp directory detected, and no model tag specified')
                return []
            else: # no tmp folder, target usertag provided
                if not os.path.isdir('{}/{}'.format(result_path, args.tag)):
                    print('Model {} not found'.format(args.tag))
                    return []
                commands.append('mv {}/{} {}/tmp'.format(result_path, args.tag, src_path))


        elif args.op == 'train': # Run training op

            if os.path.isdir('./curiosity/src/tmp'): # if a tmp file exists, resume training
                inp = raw_input('tmp directory present -> resume training? (Y/N): ')
                if inp == 'N': 
                    return []
                elif inp == 'Y':
                    # get usertag from tmp/usertag.txt
                    with open('./curiosity/src/tmp/usertag.txt', 'r') as file: usertag = file.read()

                    # grab parameters from db
                    train = TrainingParams()
                    exp = ExperimentParams(train, usertag=usertag)
                    doc = train.find_by_id(exp.default_params()['params_hash'])
                    params = dict_to_command(doc.next())
                    params = params.replace('_', '-') # remember to replace '_' with '-'

                    # create training command (directory changes before this is run)
                    commands.append('{} train.py {}'.format(py_cmd, params))
                    print('Restarting existing training session')

            else: # if no tmp file, start a new training op
                # define usertag
                if 'very' in args.env_id.lower(): setting='verySparse'
                elif 'sparse' in args.env_id.lower(): setting='sparse'
                else: setting='dense'
                algo = 'icm_{}'.format(setting) # alg_setting
                trial_number = str(int(log_data[lines[algo]].strip()[-1])+1) # increment exp count
                usertag = '{}_{}'.format(algo, trial_number) # alg_setting_number

                # override arguments
                arg_set = set(vars(args))
                parameters = dict()
                for arg in vars(args):
                    if arg in training_args: parameters[arg] = getattr(args, arg)

                # store parameters in database
                trainParams = TrainingParams(parameters)
                if not args.dry_run_highlvl:
                    expParams = ExperimentParams(trainParams, usertag=usertag)
                    storage_path = expParams.get_paths()

                # create training command
                cmd = '{} train.py {}'.format(py_cmd, dict_to_command(trainParams.default_params())) # directory changes before this is run
                commands.append(cmd)

                # read central registry to create new usertag: alg_setting_# eg: icm_sparse_4, ndigo_dense_4
                if not args.dry_run_highlvl:
                    log_entry = '{} | {} | experiment_id: {} | training_params_id: {} \n'.format(datetime.now(pytz.timezone('US/Eastern')), \
                        usertag, storage_path['exp'], expParams.default_params()['params_hash'])
                    replace_in_file('experiment_log.txt', algo, trial_number) # increment count
                    registry.write(log_entry)
                    print('Experiment registered')

                    # create a usertag.txt file in tmp with the correct usertag and datetime
                    with open('./curiosity/src/usertag.txt', 'w+') as file: file.write('{}'.format(usertag))
                    commands.append('mv usertag.txt ./tmp/usertag.txt')
                    print('Starting new training session')


        elif args.op == 'demo': # Run demo op 

            demo_path = '../results/icm'
            result_path = './curiosity/results/icm'

            if not args.tag: # no usertag specified
                print('No model tag specified')
                return []

            if not os.path.isdir('{}/{}'.format(result_path, args.tag)):
                print('Model {} not found'.format(args.tag))
                return []

            params = { # default demonstration parameters
                'ckpt':'{}/{}/model/train'.format(demo_path, args.tag),
                'outdir':'{}/{}/{}'.format(demo_path, args.tag, 'demo'),
                'env-id':'doom',
                'record':True,
                'render':True,
                'num-episodes':2,
                'greedy':False,
                'random':False
            }

            # override default arguments
            arg_set = set(vars(args))
            for arg in vars(args):
                if arg in params: params[arg] = getattr(args, arg)

            # generate demo command
            cmd = '{} demo.py {}'.format(py_cmd, dict_to_command(params))
            commands.append(cmd)
            print('Starting demo with model {} on env {}'.format(args.tag, params['env-id']))

    return commands

# Run generated commands
def run():
    args = parser.parse_args()
    commands = '\n'.join(generate_commands(args))

    if len(commands) == 0: return

    if args.dry_run_highlvl: # print commands without running them
        print('Generated commands: ')
        print('-'*50)
        print(commands)
    else: 
        if args.op != 'swap': os.chdir('./curiosity/src')
        os.system(commands)

if __name__ == '__main__':
    run()



