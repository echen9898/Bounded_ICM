
from datetime import datetime
import pytz


flatten = lambda l: [item for sublist in l for item in sublist]
log_entry = lambda time, usertag, repeat, exp_id, params_id: \
            '{} | {} | {} | experiment_id: {} | params_id: {} \n' \
            .format(time, usertag, repeat, exp_id, params_id)

def get_line_index(query, file_path):
    ''' Return line numbers where a specific query was found '''
    with open(file_path, 'r') as file: lines = file.readlines()
    return [i for i, l in enumerate(lines) if query in l]

def write_to_line(entry, index, file_path):
    ''' Write a line to a .txt file at the specified index '''
    with open(file_path, 'r') as file: lines = file.readlines()
    lines[index] = entry
    with open(file_path, 'w') as file: file.write(''.join(lines))

def get_value(usertag, key, file_path):
    ''' Looks through registry file to find key value pairs
    of the form ' | key: value | ', and returns a value for
    the corresponding usertag entry
    '''
    try: index = get_line_index(usertag, file_path)[0]
    except IndexError: return None
    with open(file_path, 'r') as file: lines = file.readlines()
    sections = lines[index].split('|')
    pair = [sec.split(':') for sec in sections if key in sec]
    return flatten(pair)[-1].strip()

def create_usertag(args):
    ''' Create a usertag with the following format: alg_setting_# (e.g. icm_dense_3) '''
    with open(args.registry, 'r') as registry:
        registry_text = registry.readlines()

        # Algorithm choice (none, icm, icmpix)
        if args.unsup == None: algo = 'none'
        elif args.unsup == 'action': algo = 'icm'
        elif 'state' in args.unsup.lower(): algo = 'icmpix'

        # Reward setting (dense, sparse, verySparse)
        if 'doom' in args.env_id.lower():
            if 'very' in args.env_id.lower(): setting='verySparse'
            elif 'sparse' in args.env_id.lower(): setting='sparse'
            else: setting='dense'
        elif 'mario' in args.env_id.lower():
            setting = args.env_id

        # Unique count id
        index = get_line_index('{}_{} = '.format(algo, setting), args.registry)[0]
        trial_number = str(int(registry_text[index].strip()[-1]) + 1) 
        usertag = '{}_{}_{}'.format(algo, setting, trial_number)
        return usertag

def dict_to_command(args, store_true_args, default_params, mode):
    ''' Convert a dictionary into a sequence of command line arguments '''
    cmd = ''
    for argument in args:
        value = args[argument]
        if argument not in default_params:
            continue
        if argument in store_true_args:
            if value == True: cmd += '--{} '.format(argument.replace('_', '-'))
            continue
        cmd += '--{} {} '.format(argument.replace('_', '-'), value)
    return cmd

def update_experiment_count(filename, usertag):
    ''' Update the experiment counter in the registry file '''
    usertag_pieces = usertag.split('_')
    trial_number = int(usertag_pieces.pop(-1)) # increment the count by one
    alg_and_setting = '_'.join(usertag_pieces)
    index = get_line_index(alg_and_setting, filename)[0]
    write_to_line('{} = {} \n'.format(alg_and_setting, trial_number), index, filename)

def update_registry(args, params, usertag, num_repeats, exp_id, params_id):
    ''' Store parameters in database, and update the experiment registry '''

    # register the experiment in the registry
    update_experiment_count(args.registry, usertag)

    time = datetime.now(pytz.timezone('US/Eastern'))
    new_entry = log_entry(time, usertag, 'repeats: {}'.format(num_repeats), exp_id, params_id)
    try:
        index = get_line_index(usertag, args.registry)[0]
        write_to_line(new_entry, index, args.registry)
    except IndexError:
        with open(args.registry, 'a') as registry: registry.write(new_entry)

    # create a usertag.txt file in tmp with the correct usertag
    with open('./curiosity/src/tmp/usertag.txt', 'w+') as file:
        file.write('{}'.format(usertag))

def numeric(chars):
    ''' Returns true if string is numeric, else false '''
    try:
        float(chars)
        return True
    except ValueError:
        return False






