
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
COLORS = [ # all available colors
    '#23282A', # black
    '#a200ff', # purple
    '#00aedb', # turquoise
    '#f47835', # orange
    '#dd93d6', # pink
    '#d41243', # red
    '#326ada', # blue
    '#008000', # green
    '#b0e0e6', # light blue
    '#964019', # brown
    '#035e7e', # navy blue
    '#82644c', # light brown
    '#f52e57', # hot pink
    '#19d120', # bright green
    '#f24e1b', # bright orange
]


# ------------------------------------------- ARGUMENTS ------------------------------------------- #
parser = argparse.ArgumentParser(description='Run plotting script')

parser.add_argument('--plot-tags', default=None, help='Which operation to run: swap, train, demo, or plot')
parser.add_argument('--output-dir', default=None, help='Usertag directory where you want to save plots')
parser.add_argument('--x-axis', default='step', help='The x axis value')
parser.add_argument('--y-axis', default='global/episode_reward', help='The scalar value being plotted')
parser.add_argument('--ave-runs', default=True, help='Whether or not each plot tag should plot train_0, or all runs averaged')
parser.add_argument('--ave-tags', default=False, help='Whether or not specified plot tags should be plotted individually, or averaged')
parser.add_argument('--x-increment', type=int, default=10000, help='Spacing between x-axis values. Only used when averaging multiple curves.')
parser.add_argument('--left-x', type=int, default=0, help='Leftmost x-axis value (timestep usually).')
parser.add_argument('--right-x', type=int, default=10000000, help='Rightmost x-axis value (timestep usually).')
parser.add_argument('--span', type=float, default=150.0, help='Smoothing parameter for EWMA. 150 for Doom, 500 for Mario is safe.')
parser.add_argument('--histogram', default=False, help='Whether or not youre plotting reward histogram')

# ------------------------------------------- STYLING ------------------------------------------- #
def print_groups(groupobject):
        for name, group in groupobject:
            print('NAME: ', name)
            print(group)
            print('-'*50)

def get_new_labels():
    ''' Dynamically reassign more concise labels based on x axis units
    '''
    ax = plt.gca()
    ax.get_xaxis().get_major_formatter().set_scientific(False)
    plt.savefig('temp.png') # need to save the figure before accessing tick labels
    new_labels = list()
    for l in ax.get_xticklabels():
        val = float(l.get_text().replace('\U00002212', '-'))
        if val > 100000 and abs(val)%100000000 == val: # millions
            new_labels.append('{}M'.format(val/1000000))
        elif val > 100 and abs(val)%100000 == val: # thousands
            new_labels.append('{}k'.format(val/1000))
        else: # hundreds, or already refactored (by matplotlib)
            new_labels.append(val)
    os.system('rm temp.png')
    return new_labels

def style(args):
    fig = plt.gcf()
    ax = plt.gca()
    sb.set(font='sans')
    if args.histogram != False:
        plt.xlabel('Intrinsic Rewards')
    else:
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
    x_vals = np.arange(args.left_x, args.right_x, args.x_increment)

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
            if args.histogram in {'True', True}:
                return ea.Histograms(args.y_axis)[0]
            else:
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

    # Set colors here
    sb.set_palette(sb.blend_palette(COLORS, n_colors=len(COLORS)))

    # Extract a dataframe for each curve
    tags = args.plot_tags.split('+')
    tags = [t.strip() for t in tags]
    count = 0
    tag_frames = list()
    for usertag in tags:

        if 'mario' in usertag.lower(): os.chdir(RESULTS_PATH_MARIO)
        else: os.chdir(RESULTS_PATH_DOOM)

        # If usertag directory doesn't exist
        if not os.path.isdir(usertag):
            print('---- Model {} not found'.format(usertag))
            return

        # Extract raw data
        os.chdir('{}/model'.format(usertag))
        if args.histogram in {'True', True}:
            hist = extract_data(usertag, tags, args).histogram_value
            sb.lineplot(x=hist.bucket_limit, y=hist.bucket)
        else:
            df = extract_data(usertag, tags, args) # either averaged, or train_0
        
            tag_frames.append(df) # save to calculate std

            # Plot each curve
            if args.ave_tags in {False, 'False'}:
                plot = sb.lineplot(x=args.x_axis, y='value', data=df, label='{}'.format(usertag.replace('_', ' ')))
            # or ... concatenate them and plot at the end
            else:
                df = interpolate(df, args) # costly, but sets each curve to the same x ticks
                if count == 0: final_df = df
                else: final_df = pd.concat((final_df, df))
                count += 1
            os.chdir('../'*6)

    # If averaging over tags, plot them here at the end (with standard deviation bars)
    if args.ave_tags in {True, 'True'}:
        final_df = final_df.groupby(level=0) # group by interpolated step
        df_mean = final_df.aggregate(np.mean)
        df_std = final_df.aggregate(np.std)
        df_mean['upper'] = df_mean['value'] + df_std['value']
        df_mean['lower'] = df_mean['value'] - df_std['value']
        df.loc[df_mean['upper'] < 0] = 0 # so error bands stay above min (BUG: DOESNT ASSIGN)
        df.loc[df_mean['lower'] < 0] = 0
        df.loc[df_mean['upper'] > 1.0] = 1.0 # so error bands stay below max (DOESNT ASSIGN)
        df.loc[df_mean['lower'] > 1.0] = 1.0
        plot = sb.lineplot(x=args.x_axis, y='value', data=df_mean, label='{}'.format('Averaged tags: {}'.format(tags)))
        plot.fill_between(x=np.arange(args.left_x, args.right_x, args.x_increment), y1=df_mean['lower'], y2=df_mean['upper'], alpha=0.2)

    # Save plot option
    style(args)
    plt.show()
    if sys.version_info[0] == 2: user_inp = raw_input('SAVE PLOT? -> (Y/N): ')
    elif sys.version_info[0] == 3: user_inp = input('SAVE PLOT? -> (Y/N): ')
    if user_inp == 'Y':
        print('---- Saving plot in {} directory'.format(args.output_dir))
        if args.ave_tags in {True, 'True'}: savename = '{}_averaged_tags.png'.format('_'.join([tags[0], args.y_axis.split('/')[-1]]))
        elif args.ave_runs in {True, 'True'}: savename = '{}_averaged_runs.png'.format('_'.join([tags[0], args.y_axis.split('/')[-1]]))
        else: savename = '{}.png'.format('_'.join([tags[0], args.y_axis.split('/')[-1]]))
        os.chdir('./curiosity/results/icm/{}/{}/plots/'.format(mode, args.output_dir))
        plot.get_figure().savefig(savename)
    else:
        print('---- Skipping this one: done plotting')
        return


if __name__ == '__main__':
    args = parser.parse_args()
    plot_tags(args)





















