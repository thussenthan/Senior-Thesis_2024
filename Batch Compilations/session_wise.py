#%% import libraries
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as mpatches
import pandas as pd
import glob
from scipy.stats import ttest_ind_from_stats
from statsmodels.stats.multitest import multipletests
#%%
def is_lcop_allowed(filepath):
    # List of disallowed substrings
        disallowed_lcop_numbers = ['lcop07', 'lcop08']

        # Check if any disallowed substrings are in the filepath
        for disallowed_lcop in disallowed_lcop_numbers:
            if disallowed_lcop in filepath:
                return False  # Return False if a disallowed substring is found
        return True  # Return True if none of the disallowed substrings are found
#%%
for j in range(7, 20):
    mean_nonopto, err_nonopto, time_nonopto = [], [], []
    mean_opto, err_opto, time_opto = [], [], []
    nonopto, opto = 0, 0
        
    root_dir = '/Volumes/wang/Junuk/eyeblink/2023_ebc_pcp_ai27/'

    pattern = f'202307??_ebc{j}/lcop??light/lcop??light_nonopto_processed_data0.npy.npz'
    pattern_opto = f'202307??_ebc{j}/lcop??light/lcop??light_processed_data0.npy.npz'
    pattern_trials = f'202307??_ebc{j}/lcop??light/lcop??light_data.csv'

    # Use glob to find all files matching the pattern, considering the specific directory structure
    file_paths = glob.glob(f'{root_dir}{pattern}', recursive=True)
    file_paths_opto = glob.glob(f'{root_dir}{pattern_opto}', recursive=True)
    file_paths_trials = glob.glob(f'{root_dir}{pattern_trials}', recursive=True)

    # Filter file paths to exclude disallowed lcop numbers
    file_paths = [path for path in file_paths if is_lcop_allowed(path)]
    file_paths_opto = [path for path in file_paths_opto if is_lcop_allowed(path)]
    file_paths_trials = [path for path in file_paths_trials if is_lcop_allowed(path)]

    for file_path in file_paths:
        with np.load(file_path) as data:
            mean_nonopto.append(data['mean'])
            err_nonopto.append(data['std'])
            time_nonopto.append(data['time'])

    batch_mean_nonopto = np.mean(np.stack(mean_nonopto), axis=0)
    batch_err_nonopto = np.mean(np.stack(err_nonopto), axis=0)
    batch_time_nonopto = np.mean(np.stack(time_nonopto), axis=0)

    for file_path_opto in file_paths_opto:
        with np.load(file_path_opto) as data_opto:
            mean_opto.append(data_opto['mean'])
            err_opto.append(data_opto['std'])
            time_opto.append(data_opto['time'])

    batch_mean_opto = np.mean(np.stack(mean_opto), axis=0)
    batch_err_opto = np.mean(np.stack(err_opto), axis=0)
    batch_time_opto = np.mean(np.stack(time_opto), axis=0)

    for file_path_trial in file_paths_trials:
        data_trial = pd.read_csv(file_path_trial)
        nonopto += sum(data_trial.kind==2)
        opto += sum(data_trial.kind==5)

    plt.plot(batch_time_nonopto, batch_mean_nonopto, color='b')
    plt.fill_between(batch_time_nonopto, batch_mean_nonopto-batch_err_nonopto, batch_mean_nonopto+batch_err_nonopto, alpha=.1, color='k', lw=0)
    plt.plot(batch_time_opto, batch_mean_opto, color='r')
    plt.fill_between(batch_time_opto, batch_mean_opto-batch_err_opto, batch_mean_opto+batch_err_opto, alpha=.1, color='k', lw=0)
    blue_patch = mpatches.Patch(color='b', label=f'non-OPTO ({nonopto})')
    red_patch = mpatches.Patch(color='r', label=f'OPTO ({opto})')
    plt.legend(handles=[blue_patch, red_patch], loc='upper left')

    n,x = plt.ylim()
    plt.vlines(1000, n, x, linestyle='--', color='grey')
    plt.vlines(1250., n, x, linestyle='--', color='grey')


    plt.title('EBC Day {}'.format(j))
    plt.xlabel('Time from CS (msec)')
    plt.ylabel('Eyelid (a.u.)')
    plt.savefig(f'/Users/thussenthanwalter-angelo/Desktop/Thesis/Session Wise/ebc{j}.png')
    plt.show()

    plt.plot(batch_time_nonopto, batch_mean_nonopto, color='b')
    plt.fill_between(batch_time_nonopto, batch_mean_nonopto-batch_err_nonopto, batch_mean_nonopto+batch_err_nonopto, alpha=.1, color='k', lw=0)
    plt.plot(batch_time_opto, batch_mean_opto, color='r')
    plt.fill_between(batch_time_opto, batch_mean_opto-batch_err_opto, batch_mean_opto+batch_err_opto, alpha=.1, color='k', lw=0)
    blue_patch = mpatches.Patch(color='b', label=f'non-OPTO ({nonopto})')
    red_patch = mpatches.Patch(color='r', label=f'OPTO ({opto})')
    plt.legend(handles=[blue_patch, red_patch], loc='upper left')

    n,x = plt.ylim()
    plt.vlines(1000, n, x, linestyle='--', color='grey')
    plt.vlines(1250, n, x, linestyle='--', color='grey')

    plt.title('EBC Day {}'.format(j))
    plt.xlabel('Time from CS (msec)')
    plt.ylabel('Eyelid (a.u.)')
    plt.xlim(900, 1500)
    plt.savefig(f'/Users/thussenthanwalter-angelo/Desktop/Thesis/Session Wise/ebc{j}_zoomed.png')
    plt.show()

    
    if len(batch_mean_nonopto) != len(batch_mean_opto):
        raise ValueError("Arrays must be of the same length")

    t_stat, p_values = ttest_ind_from_stats(batch_mean_nonopto, batch_err_nonopto, nonopto, batch_mean_opto, batch_err_opto, opto)

    p_adjusted = multipletests(p_values, method='bonferroni')
    p_adjusted = p_adjusted[1]

    plt.plot(batch_time_nonopto, batch_mean_nonopto, color='b')
    plt.fill_between(batch_time_nonopto, batch_mean_nonopto-batch_err_nonopto, batch_mean_nonopto+batch_err_nonopto, alpha=.1, color='k', lw=0)
    plt.plot(batch_time_opto, batch_mean_opto, color='r')
    plt.fill_between(batch_time_opto, batch_mean_opto-batch_err_opto, batch_mean_opto+batch_err_opto, alpha=.1, color='k', lw=0)
    blue_patch = mpatches.Patch(color='b', label=f'non-OPTO ({nonopto})')
    red_patch = mpatches.Patch(color='r', label=f'OPTO ({opto})')
    plt.legend(handles=[blue_patch, red_patch], loc='upper left')

    mean_time = (batch_time_nonopto + batch_time_opto) / 2
    indices = [index for index, value in enumerate(mean_time) if 1000 <= value <= 1500]

    # Access current axes and figure
    ax = plt.gca()
    fig = plt.gcf()

    # Ensure the renderer is initialized
    fig.canvas.draw()

    # Get the height of the axes in pixels
    bbox = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    axes_height_in_pixels = bbox.height * fig.dpi

    # Calculate the y data coordinate for 90% height in the axes
    y_95_percent = ax.transData.inverted().transform((0, 0.95 * axes_height_in_pixels))[1]

    # Adding significance markers
    for i in indices:
        # Significance level alpha = 0.05 for '*'
        if p_adjusted[i] < 0.05:
            plt.annotate('*', (mean_time[i], y_95_percent), textcoords="offset points", xytext=(0, 10), ha='center', color='red')
        # Significance level alpha = 0.01 for '**'
        if p_adjusted[i] < 0.01:
            plt.annotate('*', (mean_time[i], y_95_percent), textcoords="offset points", xytext=(0, 5), ha='center', color='red')
        # Significance level alpha = 0.001 for '***'
        if p_adjusted[i] < 0.001:
            plt.annotate('*', (mean_time[i], y_95_percent), textcoords="offset points", xytext=(0, 0), ha='center', color='red')
            
    n,x = plt.ylim()
    plt.vlines(1000, n, x, linestyle='--', color='grey')
    plt.vlines(1250., n, x, linestyle='--', color='grey')
    plt.xlim(900, 1500)

    plt.title(f'EBC Day {j} Significance Comparison of One Second Opto Stim and Non-Opto Trials')
    plt.xlabel('Time from CS (msec)')
    plt.ylabel('Eyelid (a.u.)')
    plt.savefig(f'/Users/thussenthanwalter-angelo/Desktop/Thesis/Session Wise/ebc{j}_significance.png')
    plt.show()

#%%
for j in range(22, 29):
    mean_nonopto, err_nonopto, time_nonopto = [], [], []
    mean_opto, err_opto, time_opto = [], [], []
    nonopto, opto = 0, 0
        
    root_dir = '/Volumes/wang/Junuk/eyeblink/2023_ebc_pcp_ai27/'
    
    pattern = f'202307??_ebc{j}/lcop??light/lcop??light_nonopto_processed_data0.npy.npz'
    pattern_opto = f'202307??_ebc{j}/lcop??light/lcop??light_processed_data0.npy.npz'
    pattern_trials = f'202307??_ebc{j}/lcop??light/lcop??light_data.csv'

    # Use glob to find all files matching the pattern, considering the specific directory structure
    file_paths = glob.glob(f'{root_dir}{pattern}', recursive=True)
    file_paths_opto = glob.glob(f'{root_dir}{pattern_opto}', recursive=True)
    file_paths_trials = glob.glob(f'{root_dir}{pattern_trials}', recursive=True)

    # Filter file paths to exclude disallowed lcop numbers
    file_paths = [path for path in file_paths if is_lcop_allowed(path)]
    file_paths_opto = [path for path in file_paths_opto if is_lcop_allowed(path)]
    file_paths_trials = [path for path in file_paths_trials if is_lcop_allowed(path)]

    for file_path in file_paths:
        with np.load(file_path) as data:
            mean_nonopto.append(data['mean'])
            err_nonopto.append(data['std'])
            time_nonopto.append(data['time'])

    batch_mean_nonopto = np.mean(np.stack(mean_nonopto), axis=0)
    batch_err_nonopto = np.mean(np.stack(err_nonopto), axis=0)
    batch_time_nonopto = np.mean(np.stack(time_nonopto), axis=0)

    for file_path_opto in file_paths_opto:
        with np.load(file_path_opto) as data_opto:
            mean_opto.append(data_opto['mean'])
            err_opto.append(data_opto['std'])
            time_opto.append(data_opto['time'])

    batch_mean_opto = np.mean(np.stack(mean_opto), axis=0)
    batch_err_opto = np.mean(np.stack(err_opto), axis=0)
    batch_time_opto = np.mean(np.stack(time_opto), axis=0)

    for file_path_trial in file_paths_trials:
        data_trial = pd.read_csv(file_path_trial)
        nonopto += sum(data_trial.kind==2)
        opto += sum(data_trial.kind==5)

    plt.plot(batch_time_nonopto, batch_mean_nonopto, color='b')
    plt.fill_between(batch_time_nonopto, batch_mean_nonopto-batch_err_nonopto, batch_mean_nonopto+batch_err_nonopto, alpha=.1, color='k', lw=0)
    plt.plot(batch_time_opto, batch_mean_opto, color='r')
    plt.fill_between(batch_time_opto, batch_mean_opto-batch_err_opto, batch_mean_opto+batch_err_opto, alpha=.1, color='k', lw=0)
    blue_patch = mpatches.Patch(color='b', label=f'non-OPTO ({nonopto})')
    red_patch = mpatches.Patch(color='r', label=f'OPTO ({opto})')
    plt.legend(handles=[blue_patch, red_patch], loc='upper left')

    n,x = plt.ylim()
    plt.vlines(2000, n, x, linestyle='--', color='grey')
    plt.vlines(2250., n, x, linestyle='--', color='grey')


    plt.title('EBC Day {}'.format(j))
    plt.xlabel('Time from CS (msec)')
    plt.ylabel('Eyelid (a.u.)')
    plt.savefig(f'/Users/thussenthanwalter-angelo/Desktop/Thesis/Session Wise/ebc{j}.png')
    plt.show()

    plt.plot(batch_time_nonopto, batch_mean_nonopto, color='b')
    plt.fill_between(batch_time_nonopto, batch_mean_nonopto-batch_err_nonopto, batch_mean_nonopto+batch_err_nonopto, alpha=.1, color='k', lw=0)
    plt.plot(batch_time_opto, batch_mean_opto, color='r')
    plt.fill_between(batch_time_opto, batch_mean_opto-batch_err_opto, batch_mean_opto+batch_err_opto, alpha=.1, color='k', lw=0)
    blue_patch = mpatches.Patch(color='b', label=f'non-OPTO ({nonopto})')
    red_patch = mpatches.Patch(color='r', label=f'OPTO ({opto})')
    plt.legend(handles=[blue_patch, red_patch], loc='upper left')

    n,x = plt.ylim()
    plt.vlines(2000, n, x, linestyle='--', color='grey')
    plt.vlines(2250, n, x, linestyle='--', color='grey')

    plt.title('EBC Day {}'.format(j))
    plt.xlabel('Time from CS (msec)')
    plt.ylabel('Eyelid (a.u.)')
    plt.xlim(1900, 2500)
    plt.savefig(f'/Users/thussenthanwalter-angelo/Desktop/Thesis/Session Wise/ebc{j}_zoomed.png')
    plt.show()

    
    if len(batch_mean_nonopto) != len(batch_mean_opto):
        raise ValueError("Arrays must be of the same length")

    t_stat, p_values = ttest_ind_from_stats(batch_mean_nonopto, batch_err_nonopto, nonopto, batch_mean_opto, batch_err_opto, opto)

    p_adjusted = multipletests(p_values, method='bonferroni')
    p_adjusted = p_adjusted[1]

    plt.plot(batch_time_nonopto, batch_mean_nonopto, color='b')
    plt.fill_between(batch_time_nonopto, batch_mean_nonopto-batch_err_nonopto, batch_mean_nonopto+batch_err_nonopto, alpha=.1, color='k', lw=0)
    plt.plot(batch_time_opto, batch_mean_opto, color='r')
    plt.fill_between(batch_time_opto, batch_mean_opto-batch_err_opto, batch_mean_opto+batch_err_opto, alpha=.1, color='k', lw=0)
    blue_patch = mpatches.Patch(color='b', label=f'non-OPTO ({nonopto})')
    red_patch = mpatches.Patch(color='r', label=f'OPTO ({opto})')
    plt.legend(handles=[blue_patch, red_patch], loc='upper left')

    mean_time = (batch_time_nonopto + batch_time_opto) / 2
    indices = [index for index, value in enumerate(mean_time) if 2000 <= value <= 2500]

    # Access current axes and figure
    ax = plt.gca()
    fig = plt.gcf()

    # Ensure the renderer is initialized
    fig.canvas.draw()

    # Get the height of the axes in pixels
    bbox = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    axes_height_in_pixels = bbox.height * fig.dpi

    # Calculate the y data coordinate for 90% height in the axes
    y_95_percent = ax.transData.inverted().transform((0, 0.95 * axes_height_in_pixels))[1]

    # Adding significance markers
    for i in indices:
        # Significance level alpha = 0.05 for '*'
        if p_adjusted[i] < 0.05:
            plt.annotate('*', (mean_time[i], y_95_percent), textcoords="offset points", xytext=(0, 10), ha='center', color='red')
        # Significance level alpha = 0.01 for '**'
        if p_adjusted[i] < 0.01:
            plt.annotate('*', (mean_time[i], y_95_percent), textcoords="offset points", xytext=(0, 5), ha='center', color='red')
        # Significance level alpha = 0.001 for '***'
        if p_adjusted[i] < 0.001:
            plt.annotate('*', (mean_time[i], y_95_percent), textcoords="offset points", xytext=(0, 0), ha='center', color='red')
            
    n,x = plt.ylim()
    plt.vlines(2000, n, x, linestyle='--', color='grey')
    plt.vlines(2250., n, x, linestyle='--', color='grey')
    plt.xlim(1900, 2500)

    plt.title(f'EBC Day {j} Significance Comparison of Two Second Opto Stim and Non-Opto Trials')
    plt.xlabel('Time from CS (msec)')
    plt.ylabel('Eyelid (a.u.)')
    plt.savefig(f'/Users/thussenthanwalter-angelo/Desktop/Thesis/Session Wise/ebc{j}_significance.png')
    plt.show()

# %%
