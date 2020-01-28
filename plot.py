
import os, sys
import subprocess
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
from tensorboard.backend.event_processing import event_accumulator

# ------------------------------------------- DEFAULTS ------------------------------------------- #
RESULTS_PATH_DOOM = './curiosity/results/icm/doom'
RESULTS_PATH_MARIO = './curiosity/results/icm/mario'


# ------------------------------------------- ARGUMENTS ------------------------------------------- #
parser = argparse.ArgumentParser(description='Run plotting script')

parser.add_argument('--plot-tags', default=None, help='Which operation to run: swap, train, demo, or plot')
parser.add_argument('--output-dir', default=None, help='Usertag directory where you want to save plots')
parser.add_argument('--x-axis', default='step', help='The x axis value')
parser.add_argument('--y-axis', default='global/episode_reward', help='The scalar value being plotted')
parser.add_argument('--ave-runs', default=True, help='Whether or not each plot tag should plot train_0, or all runs averaged')
parser.add_argument('--ave-tags', default=False, help='Whether or not specified plot tags should be plotted individually, or averaged')
parser.add_argument('--x-increment', default=10000, help='Spacing between x-axis values. Only used when averaging multiple curves.')
parser.add_argument('--left-x', default=0, help='Leftmost x-axis value (timestep usually).')
parser.add_argument('--right-x', default=10000000, help='Rightmost x-axis value (timestep usually).')
parser.add_argument('--span', type=float, default=150.0, help='Smoothing parameter for EWMA. 150 for Doom, 500 for Mario is safe.')

# ------------------------------------------- STYLING ------------------------------------------- #
def get_new_labels():
    ''' Dynamically reassign more concise labels based on x axis units
    '''
    ax = plt.gca()
    ax.get_xaxis().get_major_formatter().set_scientific(False)
    plt.savefig('temp.jpg') # need to save the figure before accessing tick labels
    new_labels = list()
    for l in ax.get_xticklabels():
        val = float(l.get_text().replace('\U00002212', '-'))
        if val > 100000 and abs(val)%100000000 == val: # millions
            new_labels.append('{}M'.format(val/1000000))
        elif val > 100 and abs(val)%100000 == val: # thousands
            new_labels.append('{}k'.format(val/1000))
        else: # hundreds, or already refactored (by matplotlib)
            new_labels.append(val)
    os.system('rm temp.jpg')
    return new_labels

def style(args):
    fig = plt.gcf()
    ax = plt.gca()
    sb.set(font='sans')
    plt.xlabel('Training Steps (millions)')
    y_label = args.y_axis.split('/')[-1]
    y_label = ' '.join(y_label.split('_')).capitalize()
    plt.ylabel(y_label)
    ax.set_xticklabels(get_new_labels())


# ------------------------------------------- UTILITIES ------------------------------------------- #
def interpolate(df, args):
    ''' Takes in a dataframe, and interpolates values for new x values based
    on arguments (x_axis, y_axis).
    '''

    # FOLD THIS INTO ARGS
    x_vals = np.arange(int(args.left_x), int(args.right_x), int(args.x_increment))

    values = list()
    for x in x_vals:
        # The following sort of complicated line finds the nearest existing upper/lower bound to x
        # and then averages their values and appends it
        values.append(df.iloc[(df[args.x_axis]-x).abs().argsort()[:2]].mean()['value'])

    return pd.DataFrame({'step':list(x_vals), 'value':values})


def extract_data(usertag, tags, args):
    ''' Reads in a usertag specifying the model to extract from. Extracts
    a csv file for each run by concatenating all events files for the .
    If multiple events files are present, concatenate them in sorted order by 
    x_axis ('step' by default) to generate overall file. y_axis specifies the
    scalar value being extracted.
    '''

    # No scalar value to extract specified
    if args.y_axis is None:
        print('---- No scalar value specified')
        return

    # Extract csv's for all runs
    train_dirs = [d for d in os.listdir() if 'train_' in d]
    if args.ave_runs in {'False', False}:
        train_dirs = ['train_0']
    worker_dfs = list() # dataframes for each worker (train_0, train_1, etc)
    count = 0
    for d in train_dirs:
        os.chdir(d)
        events = [file for file in os.listdir() if 'events.out.tfevents' in file]
        event_frames = list() # dataframe for each event file (to be stitched together)
        for event in events:
            ea = event_accumulator.EventAccumulator(event, size_guidance={event_accumulator.SCALARS: 0})
            ea.Reload() # need to call this everytime before loading data
            print(ea.Tags()) # check available tags
            try: event_frames.append(pd.DataFrame(ea.Scalars(args.y_axis)))
            except KeyError: continue
        event_frames = sorted(event_frames, key=lambda df: df[args.x_axis][0])

        # Save dataframe for this worker
        filename = '_'.join([usertag, args.y_axis.split('/')[-1], d.split('_')[-1]])
        print('Extracted {}'.format(filename))
        worker_df = pd.concat(event_frames)

        worker_df = worker_df.ewm(span=args.span).mean() # smooth curve

        # TEMPORARY:
        # worker_df.to_csv('{}.csv'.format(filename))
        # os.system('mv {}.csv ../../../{}/plots'.format(filename, args.output_dir))
        # int_df.to_csv('{}_INT.csv'.format(filename))
        # os.system('mv {}_INT.csv ../../../{}/plots'.format(filename, args.output_dir))

        os.chdir('../')

        if args.ave_runs in {False, 'False'}: return worker_df # return train_0

        # Interpolate results, and stack them
        interp_df = interpolate(worker_df, args) # interpolate on a common x-axis
        if count == 0: final_df = interp_df # start of the stack
        else: final_df = pd.concat((final_df, interp_df)) # add to the stack
        count += 1

    return final_df.groupby(level=0).mean() # group and average

# ------------------------------------------- MAIN ------------------------------------------- #
def plot_tags(args):
    ''' Usertags are passed in separated by '+' signs
    '''

    # Setup output directory
    if 'mario' in args.output_dir.strip().lower(): mode = 'mario'
    else: mode = 'doom'
    if not os.path.isdir('./curiosity/results/icm/{}/{}'.format(mode, args.output_dir.strip())):
        print('---- Output directory {} is invalid'.format(args.output_dir))
        return

    if not os.path.isdir('./{}/plots/'.format(args.output_dir)): 
        os.system('mkdir -p ./curiosity/results/icm/{}/{}/plots/'.format(mode, args.output_dir))

    # Extract a dataframe for each curve
    tags = args.plot_tags.split('+')
    tags = [t.strip() for t in tags]
    count = 0
    for usertag in tags:

        if 'mario' in usertag.lower(): os.chdir(RESULTS_PATH_MARIO)
        else: os.chdir(RESULTS_PATH_DOOM)

        # If usertag directory doesn't exist
        if not os.path.isdir(usertag):
            print('---- Model {} not found'.format(usertag))
            return

        # Extract raw data
        os.chdir('{}/model'.format(usertag))
        df = extract_data(usertag, tags, args) # either averaged, or train_0

        # Plot each curve
        if args.ave_tags in {False, 'False'}:
            plot = sb.lineplot(x=args.x_axis, y='value', data=df, label='{}'.format(usertag.replace('_', ' ')))
        # or ... concatenate them and plot at the end
        else:
            if count == 0: final_df = df
            else: final_df = pd.concat((final_df, df))
            count += 1
        os.chdir('../'*6)

    # If averaging over tags, plot them here at the end
    if args.ave_tags in {True, 'True'}:
        final_df = final_df.groupby(level=0).mean()
        plot = sb.lineplot(x=args.x_axis, y='value', data=final_df, label='{}'.format('Averaged tags: {}'.format(tags)))

    # Save plot option
    style(args)
    plt.show()
    if sys.version_info[0] == 2: user_inp = raw_input('SAVE PLOT? -> (Y/N): ')
    elif sys.version_info[0] == 3: user_inp = input('SAVE PLOT? -> (Y/N): ')
    if user_inp == 'Y':
        print('---- Saving plot in {} directory'.format(args.output_dir))
        if args.ave_tags in {True, 'True'}: savename = '{}_averaged_tags.jpg'.format('_'.join([tags[0], args.y_axis.split('/')[-1]]))
        elif args.ave_runs in {True, 'True'}: savename = '{}_averaged_runs.jpg'.format('_'.join([tags[0], args.y_axis.split('/')[-1]]))
        else: savename = '{}.jpg'.format('_'.join([tags[0], args.y_axis.split('/')[-1]]))
        os.chdir('./curiosity/results/icm/{}/{}/plots/'.format(mode, args.output_dir))
        plot.get_figure().savefig(savename)
    else:
        print('---- Skipping this one: done plotting')
        return


if __name__ == '__main__':
    args = parser.parse_args()
    plot_tags(args)





















