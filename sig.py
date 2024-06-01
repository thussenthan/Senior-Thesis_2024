#%% import libraries
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as mpatches
import pandas as pd
import glob
from scipy.stats import ttest_ind_from_stats
#%%
mylist = [1]
for j in mylist:
    if j < 10:  # Adding a leading zero if the number is less than 10
        j = f'0{j}'

    mean_nonopto = []
    err_nonopto = []
    time_nonopto = []

    mean_opto = []
    err_opto = []
    time_opto = []

    nonopto = 0
    opto = 0

    file_paths = []
    file_paths_opto = []
    file_paths_trials = []

    root_dir = '/Volumes/wang/Junuk/eyeblink/2023_ebc_pcp_ai27/'

    # Iterate over the range from ebc7 to ebc19
    for ebc_number in range(21, 30):
        pattern = f'202307??_ebc{ebc_number}/lcop{j}light/lcop{j}light_nonopto_processed_data0.npy.npz'
        pattern_opto = f'202307??_ebc{ebc_number}/lcop{j}light/lcop{j}light_processed_data0.npy.npz'
        pattern_trials = f'202307??_ebc{ebc_number}/lcop{j}light/lcop{j}light_data.csv'

        file_path_temp = glob.glob(f'{root_dir}{pattern}', recursive=True)
        file_path_opto_temp = glob.glob(f'{root_dir}{pattern_opto}', recursive=True)
        file_path_trials_temp = glob.glob(f'{root_dir}{pattern_trials}', recursive=True)

        # Add the found file paths to the list
        file_paths.extend(file_path_temp)    
        file_paths_opto.extend(file_path_opto_temp)    
        file_paths_trials.extend(file_path_trials_temp)
    
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



    if len(batch_mean_nonopto) != len(batch_mean_opto):
        raise ValueError("Arrays must be of the same length")

    t_stat, p_values = ttest_ind_from_stats(batch_mean_nonopto, batch_err_nonopto, nonopto, batch_mean_opto, batch_err_opto, opto)

    # Find significant differences
    significant_differences = p_values < alpha

    print("P-values:", p_values)
    print("Significant differences at corrected alpha level:", significant_differences)


    plt.plot(batch_time_nonopto, batch_mean_nonopto, color='b')
    plt.fill_between(batch_time_nonopto, batch_mean_nonopto-batch_err_nonopto, batch_mean_nonopto+batch_err_nonopto, alpha=.1, color='k', lw=0)
    plt.plot(batch_time_opto, batch_mean_opto, color='r')
    plt.fill_between(batch_time_opto, batch_mean_opto-batch_err_opto, batch_mean_opto+batch_err_opto, alpha=.1, color='k', lw=0)
    blue_patch = mpatches.Patch(color='b', label=f'non-OPTO ({nonopto})')
    red_patch = mpatches.Patch(color='r', label=f'OPTO ({opto})')
    plt.legend(handles=[blue_patch, red_patch], loc='upper left')

    mean_time = (batch_time_nonopto + batch_time_opto) / 2
    indices = [index for index, value in enumerate(mean_time) if 2000 <= value <= 2400]

    # Adding significance markers
    for i in indices:
        # Significance level = 0.05 for '*'
        if p_values[i] < 0.05:
            plt.annotate('*', (mean_time[i], 0.8*x), textcoords="offset points", xytext=(0, 10), ha='center', color='red')
        # Significance level = 0.01 for '**'
        if p_values[i] < 0.01:
            plt.annotate('*', (mean_time[i], 0.8*x), textcoords="offset points", xytext=(0, 5), ha='center', color='red')
        # Significance level = 0.001 for '***'
        if p_values[i] < 0.001:
            plt.annotate('*', (mean_time[i], 0.8*x), textcoords="offset points", xytext=(0, 0), ha='center', color='red')
            
    n,x = plt.ylim()
    plt.vlines(2000, n, x, linestyle='--', color='grey')
    plt.vlines(2250., n, x, linestyle='--', color='grey')
    plt.xlim(1900, 2400)

    plt.title(f'Lcop{j} Significance Comparison of Two Second Opto Stim and Non-Opto Trials')
    plt.xlabel('Time from CS (msec)')
    plt.ylabel('Eyelid (a.u.)')
    plt.show()
# %%
