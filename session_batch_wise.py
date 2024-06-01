#%% import libraries
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as mpatches
import os
import glob
from scipy.stats import ttest_rel
#%%
for j in range(22, 30):
    mean_nonopto_1sec = []
    err_nonopto_1sec = []
    time_nonopto_1sec = []

    mean_opto_1sec = []
    err_opto_1sec = []
    time_opto_1sec = []
    root_dir = '/Volumes/wang/Junuk/eyeblink/2023_ebc_pcp_ai27/'

    pattern = f'202307??_ebc{j}/lcop??light/lcop??light_nonopto_processed_data0.npy.npz'

    pattern_opto = f'202307??_ebc{j}/lcop??light/lcop??light_processed_data0.npy.npz'

    # Use glob to find all files matching the pattern, considering the specific directory structure
    file_paths = glob.glob(f'{root_dir}{pattern}', recursive=True)
    file_paths_opto = glob.glob(f'{root_dir}{pattern_opto}', recursive=True)
    print(file_paths_opto)

    # This will give you a list of all files matching the _nonopto_processed_data0.npy.npz pattern
    for file_path in file_paths:
        with np.load(file_path) as data:
            mean_nonopto_1sec.append(data['mean'])
            err_nonopto_1sec.append(data['std'])
            time_nonopto_1sec.append(data['time'])

    batch_mean_nonopto_1sec = np.mean(np.stack(mean_nonopto_1sec), axis=0)
    batch_err_nonopto_1sec = np.mean(np.stack(err_nonopto_1sec), axis=0)
    batch_time_nonopto_1sec = np.mean(np.stack(time_nonopto_1sec), axis=0)

    for file_path_opto in file_paths_opto:
        with np.load(file_path_opto) as data_opto:
            mean_opto_1sec.append(data_opto['mean'])
            err_opto_1sec.append(data_opto['std'])
            time_opto_1sec.append(data_opto['time'])

    batch_mean_opto_1sec = np.mean(np.stack(mean_opto_1sec), axis=0)
    batch_err_opto_1sec = np.mean(np.stack(err_opto_1sec), axis=0)
    batch_time_opto_1sec = np.mean(np.stack(time_opto_1sec), axis=0)

    plt.plot(batch_time_nonopto_1sec, batch_mean_nonopto_1sec, color='b')
    plt.fill_between(batch_time_nonopto_1sec, batch_mean_nonopto_1sec-batch_err_nonopto_1sec, batch_mean_nonopto_1sec+batch_err_nonopto_1sec, alpha=.1, color='k', lw=0)
    plt.plot(batch_time_opto_1sec, batch_mean_opto_1sec, color='r')
    plt.fill_between(batch_time_opto_1sec, batch_mean_opto_1sec-batch_err_opto_1sec, batch_mean_opto_1sec+batch_err_opto_1sec, alpha=.1, color='k', lw=0)
    blue_patch = mpatches.Patch(color='b', label='non-OPTO')
    red_patch = mpatches.Patch(color='r', label='OPTO')
    plt.legend(handles=[blue_patch, red_patch], loc='upper left')

    n,x = plt.ylim()
    plt.vlines(2000, n, x, linestyle='--', color='grey')
    plt.vlines(2250., n, x, linestyle='--', color='grey')

    plt.title('EBC Day {}'.format(j))
    plt.xlabel('Time from CS (msec)')
    plt.ylabel('Eyelid (a.u.)')
    plt.savefig(f'/Users/thussenthanwalter-angelo/Desktop/Thesis/Session Wise/ebc{j}.png')
    plt.show()

    plt.plot(batch_time_nonopto_1sec, batch_mean_nonopto_1sec, color='b')
    plt.fill_between(batch_time_nonopto_1sec, batch_mean_nonopto_1sec-batch_err_nonopto_1sec, batch_mean_nonopto_1sec+batch_err_nonopto_1sec, alpha=.1, color='k', lw=0)
    plt.plot(batch_time_opto_1sec, batch_mean_opto_1sec, color='r')
    plt.fill_between(batch_time_opto_1sec, batch_mean_opto_1sec-batch_err_opto_1sec, batch_mean_opto_1sec+batch_err_opto_1sec, alpha=.1, color='k', lw=0)
    blue_patch = mpatches.Patch(color='b', label='non-OPTO')
    red_patch = mpatches.Patch(color='r', label='OPTO')
    plt.legend(handles=[blue_patch, red_patch], loc='upper left')

    n,x = plt.ylim()
    plt.vlines(2000, n, x, linestyle='--', color='grey')
    plt.vlines(2250., n, x, linestyle='--', color='grey')

    plt.title('EBC Day {}'.format(j))
    plt.xlabel('Time from CS (msec)')
    plt.ylabel('Eyelid (a.u.)')
    plt.xlim(1900, 2350)
    plt.savefig(f'/Users/thussenthanwalter-angelo/Desktop/Thesis/Session Wise/ebc{j}_zoomed.png')
    plt.show()
    if len(batch_mean_nonopto_1sec) != len(batch_mean_opto_1sec):
        raise ValueError("Arrays must be of the same length")

    p_values = np.zeros(len(batch_mean_nonopto_1sec))
    for i in range(len(batch_mean_nonopto_1sec)):
        _, p_value = ttest_rel(batch_mean_nonopto_1sec[i], batch_mean_opto_1sec[i])
        p_values[i] = p_value

    # Significance level
    alpha = 0.05

    plt.plot(batch_time_nonopto_1sec, batch_mean_nonopto_1sec, color='b')
    plt.fill_between(batch_time_nonopto_1sec, batch_mean_nonopto_1sec-batch_err_nonopto_1sec, batch_mean_nonopto_1sec+batch_err_nonopto_1sec, alpha=.1, color='k', lw=0)
    plt.plot(batch_time_opto_1sec, batch_mean_opto_1sec, color='r')
    plt.fill_between(batch_time_opto_1sec, batch_mean_opto_1sec-batch_err_opto_1sec, batch_mean_opto_1sec+batch_err_opto_1sec, alpha=.1, color='k', lw=0)
    blue_patch = mpatches.Patch(color='b', label='non-OPTO')
    red_patch = mpatches.Patch(color='r', label='OPTO')
    plt.legend(handles=[blue_patch, red_patch], loc='upper left')

    mean_time = (batch_time_nonopto_1sec + batch_time_opto_1sec) / 2
    indices = [index for index, value in enumerate(mean_time) if 1900 <= value <= 2350]

    # Adding significance markers
    for i in indices:
        if p_values[i] < alpha:
            # Adjust the height for annotation slightly above the higher line
            height = max(batch_time_nonopto_1sec[i], batch_time_opto_1sec[i]) + 0.1
            plt.annotate('*', (x[i], height), textcoords="offset points", xytext=(0,5), ha='center', color='red')
            # Optional: Annotate the p-value as well
            plt.annotate(f'{p_values[i]:.3f}', (x[i], height), textcoords="offset points", xytext=(0,15), ha='center', color='blue')

    n,x = plt.ylim()
    plt.vlines(2000, n, x, linestyle='--', color='grey')
    plt.vlines(2250., n, x, linestyle='--', color='grey')
    plt.xlim(1900, 2350)

    plt.title(f'EBC Day {j} Significance Comparison of Opto and Non-Opto Trials')
    plt.xlabel('Time from CS (msec)')
    plt.ylabel('Eyelid (a.u.)')
    plt.savefig(f'/Users/thussenthanwalter-angelo/Desktop/Thesis/Session Wise/ebc{j}_significance.png')
    plt.show()
# %%
