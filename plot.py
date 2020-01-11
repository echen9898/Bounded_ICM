
import os, sys
import subprocess
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
from tensorboard.backend.event_processing import event_accumulator

# ------------------------------------------- DEFAULTS ------------------------------------------- #
RESULTS_PATH = './curiosity/results/icm'


# ------------------------------------------- ARGUMENTS ------------------------------------------- #
parser = argparse.ArgumentParser(description='Run plotting script')

parser.add_argument('--plot-tags', default=None, help='Which operation to run: swap, train, demo, or plot')
parser.add_argument('--output-dir', default=None, help='Usertag directory where you want to save plots')
parser.add_argument('--x-axis', default='step', help='The x axis value')
parser.add_argument('--y-axis', default='global/episode_reward', help='The scalar value being plotted')
parser.add_argument('--ave-runs', default=False, help='Whether or all runs should be plotted, or averaged')
# parser.add_argument('--mode', default='') # all scalars or just a single value


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
    plt.ylabel('Extrinsic Reward')
    ax.set_xticklabels(get_new_labels())


# ------------------------------------------- MAIN METHODS ------------------------------------------- #
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

    # Overlaid plots just use train_0, while all runs can be plotted for a single tag
    if len(tags) > 1: train_dirs = ['train_0']
    else: train_dirs = [d for d in os.listdir() if 'train_' in d]

    # Extract csv's for all runs
    dataframes = list() # dataframes of each run
    for d in train_dirs:
        os.chdir(d)
        events = [file for file in os.listdir() if 'events.out.tfevents' in file]
        event_frames = list() # dataframe for each event file (to be stitched together)
        for event in events:
            ea = event_accumulator.EventAccumulator(event, size_guidance={event_accumulator.SCALARS: 0})
            ea.Reload() # need to call this everytime before loading data
            try: event_frames.append(pd.DataFrame(ea.Scalars(args.y_axis)))
            except KeyError: continue
        event_frames = sorted(event_frames, key=lambda df: df[args.x_axis][0])

        # Save dataframe for this worker
        filename = '_'.join([usertag, args.y_axis.split('/')[-1], d.split('_')[-1]])
        run_df = pd.concat(event_frames)

        os.chdir('../')

        if args.ave_runs == 'False':
            return run_df # RETURN TRAIN_0
            # run_df.to_csv('{}.csv'.format(filename))
            # os.system('mv {}.csv ../../../{}/plots/data/'.format(filename, args.output_dir))
        else: 
            dataframes.append(run_df)

    if args.ave_runs == 'True':
        print('TODO: Average dataframes together, and return single dataframe')
        # pd.concat(dataframes, axis=1).to_csv('{}_averaged.csv'.format(usertag))
        # os.system('mv {}_averaged.csv ../../{}/plots/data/'.format(usertag, args.output_dir))

def plot_tags(args):
    ''' Usertags are passed in separated by '+' signs
    '''
    os.chdir(RESULTS_PATH)

    # Setup output directory
    if not os.path.isdir('./{}'.format(args.output_dir.strip())):
        print('---- Output directory {} is invalid'.format(args.output_dir))
        return

    if not os.path.isdir('./{}/plots/'.format(args.output_dir)): 
        os.system('mkdir -p ./{}/plots/'.format(args.output_dir))

    # For each tag, extract csv values, and plot
    tags = args.plot_tags.split('+')
    tags = [t.strip() for t in tags]
    for usertag in tags:

        # If usertag directory doesn't exist
        if not os.path.isdir(usertag):
            print('---- Model {} not found'.format(usertag))
            return

        # Extract raw data
        os.chdir('{}/model'.format(usertag))
        df = extract_data(usertag, tags, args) # either averaged, or train_0

        # Process data
        df = df.ewm(span=200).mean()

        # Plot
        # plot = scatter_plot(args.x_axis, 'value', df)
        plot = sb.lineplot(x=args.x_axis, y='value', data=df, label='{}'.format(usertag.replace('_', ' ')))

        os.chdir('../'*2)

    # Save plot option
    style(args)
    plt.show()
    if sys.version_info[0] == 2: user_inp = raw_input('SAVE PLOT? -> (Y/N): ')
    elif sys.version_info[0] == 3: user_inp = input('SAVE PLOT? -> (Y/N): ')
    if user_inp == 'Y':
        print('---- Saving plot in {} directory'.format(args.output_dir))
        savename = '{}.jpg'.format('_'.join([tags[0], args.y_axis.split('/')[-1]]))
        os.chdir('./{}/plots/'.format(args.output_dir))
        plot.get_figure().savefig(savename)
    else:
        print('---- Skipping this one: done plotting')
        return


# ------------------------------------------- MAIN METHOD ------------------------------------------- #
if __name__ == '__main__':
    args = parser.parse_args()
    plot_tags(args)





















