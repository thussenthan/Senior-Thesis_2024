#%% import libraries
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as mpatches
import pandas as pd
import os
import glob
from scipy.stats import ttest_ind_from_stats
from statsmodels.stats.multitest import multipletests
#%%
mean_nonopto_1sec = []
err_nonopto_1sec = []
time_nonopto_1sec = []

mean_opto_1sec = []
err_opto_1sec = []
time_opto_1sec = []

nonopto_1sec = 0
opto_1sec = 0
#%% 1 SECOND OPTO
base_path = '/Volumes/wang/Junuk/eyeblink/2023_ebc_pcp_ai27'

def is_lcop_allowed(filepath):
    # List of disallowed substrings
    disallowed_lcop_numbers = ['lcop07', 'lcop08']

    # Check if any disallowed substrings are in the filepath
    for disallowed_lcop in disallowed_lcop_numbers:
        if disallowed_lcop in filepath:
            return False  # Return False if a disallowed substring is found
    return True  # Return True if none of the disallowed substrings are found


# Create a pattern for each ebc folder from ebc6 to ebc19
ebc_patterns_1sec = [
    os.path.join(base_path, f'202307??_ebc{ebc}', 'lcop??light', '*_nonopto_processed_data0.npy.npz')
    for ebc in range(7,20)

]
ebc_patterns_opto_1sec = [
    os.path.join(base_path, f'202307??_ebc{ebc}', 'lcop??light', '*light_processed_data0.npy.npz')
    for ebc in range(7,20)

]
ebc_patterns_1sec_alltrials = [
    os.path.join(base_path, f'202307??_ebc{ebc}', 'lcop??light', 'lcop??light_data.csv')
    for ebc in range(7,20)

]

# Initialize a list to collect all matching file paths
matching_files_1sec = []
matching_files_opto_1sec = []
matching_files_1sec_alltrials = []

# Collect all files that match the ebc patterns
for pattern in ebc_patterns_1sec:
    files = glob.glob(pattern, recursive=True)
    matching_files_1sec.extend([file for file in files if is_lcop_allowed(file)])

for pattern_opto in ebc_patterns_opto_1sec:
    files = glob.glob(pattern_opto, recursive=True)
    matching_files_opto_1sec.extend([file for file in files if is_lcop_allowed(file)])

for pattern_trials in ebc_patterns_1sec_alltrials:
    files = glob.glob(pattern_trials, recursive=True)
    matching_files_1sec_alltrials.extend([file for file in files if is_lcop_allowed(file)])


#%%
# This will give you a list of all files matching the _nonopto_processed_data0.npy.npz pattern
for file_path in matching_files_1sec:
    with np.load(file_path) as data:
        mean_nonopto_1sec.append(data['mean'])
        err_nonopto_1sec.append(data['std'])
        time_nonopto_1sec.append(data['time'])

batch_mean_nonopto_1sec = np.mean(np.stack(mean_nonopto_1sec), axis=0)
batch_err_nonopto_1sec = np.mean(np.stack(err_nonopto_1sec), axis=0)
batch_time_nonopto_1sec = np.mean(np.stack(time_nonopto_1sec), axis=0)

for file_path_opto in matching_files_opto_1sec:
    with np.load(file_path_opto) as data_opto:
        mean_opto_1sec.append(data_opto['mean'])
        err_opto_1sec.append(data_opto['std'])
        time_opto_1sec.append(data_opto['time'])

batch_mean_opto_1sec = np.mean(np.stack(mean_opto_1sec), axis=0)
batch_err_opto_1sec = np.mean(np.stack(err_opto_1sec), axis=0)
batch_time_opto_1sec = np.mean(np.stack(time_opto_1sec), axis=0)

for file_path_trial in matching_files_1sec_alltrials:
    data_trial = pd.read_csv(file_path_trial)
    nonopto_1sec += sum(data_trial.kind==2)
    opto_1sec += sum(data_trial.kind==5)

#%%
plt.plot(batch_time_nonopto_1sec, batch_mean_nonopto_1sec, color='b')
plt.fill_between(batch_time_nonopto_1sec, batch_mean_nonopto_1sec-batch_err_nonopto_1sec, batch_mean_nonopto_1sec+batch_err_nonopto_1sec, alpha=.1, color='k', lw=0)
plt.plot(batch_time_opto_1sec, batch_mean_opto_1sec, color='r')
plt.fill_between(batch_time_opto_1sec, batch_mean_opto_1sec-batch_err_opto_1sec, batch_mean_opto_1sec+batch_err_opto_1sec, alpha=.1, color='k', lw=0)
blue_patch = mpatches.Patch(color='b', label=f'non-OPTO ({nonopto_1sec})')
red_patch = mpatches.Patch(color='r', label=f'OPTO ({opto_1sec})')
plt.legend(handles=[blue_patch, red_patch], loc='upper left')

n,x = plt.ylim()
plt.vlines(1000, n, x, linestyle='--', color='grey')
plt.vlines(1250., n, x, linestyle='--', color='grey')

plt.xlabel('Time from CS (msec)')
plt.ylabel('Eyelid (a.u.)')
plt.savefig(f'/Users/thussenthanwalter-angelo/Desktop/Thesis/Batch Compile/One Second Batch Compile.png')
plt.show()

plt.plot(batch_time_nonopto_1sec, batch_mean_nonopto_1sec, color='b')
plt.fill_between(batch_time_nonopto_1sec, batch_mean_nonopto_1sec-batch_err_nonopto_1sec, batch_mean_nonopto_1sec+batch_err_nonopto_1sec, alpha=.1, color='k', lw=0)
plt.plot(batch_time_opto_1sec, batch_mean_opto_1sec, color='r')
plt.fill_between(batch_time_opto_1sec, batch_mean_opto_1sec-batch_err_opto_1sec, batch_mean_opto_1sec+batch_err_opto_1sec, alpha=.1, color='k', lw=0)
blue_patch = mpatches.Patch(color='b', label=f'non-OPTO({nonopto_1sec})')
red_patch = mpatches.Patch(color='r', label=f'OPTO({opto_1sec})')
plt.legend(handles=[blue_patch, red_patch], loc='upper left')

n,x = plt.ylim()
plt.vlines(1000, n, x, linestyle='--', color='grey')
plt.vlines(1250., n, x, linestyle='--', color='grey')

plt.xlabel('Time from CS (msec)')
plt.ylabel('Eyelid (a.u.)')
plt.xlim(900, 1500)
plt.savefig(f'/Users/thussenthanwalter-angelo/Desktop/Thesis/Batch Compile/One Second Batch Compile_zoomed.png')
plt.show()
#%%
np.savez('/Users/thussenthanwalter-angelo/Desktop/Thesis/Batch Compile/batch_plot_1sec_processed_data{}.npy', 
         batch_mean_nonopto_1sec=batch_mean_nonopto_1sec,
         batch_err_nonopto_1sec=batch_err_nonopto_1sec,
         batch_time_nonopto_1sec=batch_time_nonopto_1sec,
         batch_mean_opto_1sec=batch_mean_opto_1sec, 
         batch_err_opto_1sec=batch_err_opto_1sec, 
         batch_time_opto_1sec=batch_time_opto_1sec)
#%% 2 SECOND OPTO
mean_nonopto_2sec = []
err_nonopto_2sec = []
time_nonopto_2sec = []

mean_opto_2sec = []
err_opto_2sec = []
time_opto_2sec = []

nonopto_2sec = 0
opto_2sec = 0
#%%
base_path = '/Volumes/wang/Junuk/eyeblink/2023_ebc_pcp_ai27'

def is_lcop_allowed(filepath):
    # List of disallowed substrings
    disallowed_lcop_numbers = ['lcop07', 'lcop08']

    # Check if any disallowed substrings are in the filepath
    for disallowed_lcop in disallowed_lcop_numbers:
        if disallowed_lcop in filepath:
            return False  # Return False if a disallowed substring is found
    return True  # Return True if none of the disallowed substrings are found

# Create a pattern for each ebc folder from ebc6 to ebc19
ebc_patterns_2sec = [
    os.path.join(base_path, f'202307??_ebc{ebc}', 'lcop??light', '*_nonopto_processed_data0.npy.npz')
    for ebc in range(22, 29)
]
ebc_patterns_opto_2sec = [
    os.path.join(base_path, f'202307??_ebc{ebc}', 'lcop??light', '*light_processed_data0.npy.npz')
    for ebc in range(22, 29)
]
ebc_patterns_2sec_alltrials = [
    os.path.join(base_path, f'202307??_ebc{ebc}', 'lcop??light', 'lcop??light_data.csv')
    for ebc in range(22, 29)
]
# Initialize a list to collect all matching file paths
matching_files_2sec = []
matching_files_opto_2sec = []
matching_files_2sec_alltrials = []

# Collect all files that match the ebc patterns
for pattern in ebc_patterns_2sec:
    files = glob.glob(pattern, recursive=True)
    matching_files_2sec.extend([file for file in files if is_lcop_allowed(file)])

for pattern_opto in ebc_patterns_opto_2sec:
    files = glob.glob(pattern_opto, recursive=True)
    matching_files_opto_2sec.extend([file for file in files if is_lcop_allowed(file)])

for pattern_trials in ebc_patterns_2sec_alltrials:
    files = glob.glob(pattern_trials, recursive=True)
    matching_files_2sec_alltrials.extend([file for file in files if is_lcop_allowed(file)])
#%%
# This will give you a list of all files matching the _nonopto_processed_data0.npy.npz pattern
for file_path in matching_files_2sec:
    with np.load(file_path) as data:
        mean_nonopto_2sec.append(data['mean'])
        err_nonopto_2sec.append(data['std'])
        time_nonopto_2sec.append(data['time'])

batch_mean_nonopto_2sec = np.mean(np.stack(mean_nonopto_2sec), axis=0)
batch_err_nonopto_2sec = np.mean(np.stack(err_nonopto_2sec), axis=0)
batch_time_nonopto_2sec = np.mean(np.stack(time_nonopto_2sec), axis=0)

for file_path_opto in matching_files_opto_2sec:
    with np.load(file_path_opto) as data_opto:
        mean_opto_2sec.append(data_opto['mean'])
        err_opto_2sec.append(data_opto['std'])
        time_opto_2sec.append(data_opto['time'])

batch_mean_opto_2sec = np.mean(np.stack(mean_opto_2sec), axis=0)
batch_err_opto_2sec = np.mean(np.stack(err_opto_2sec), axis=0)
batch_time_opto_2sec = np.mean(np.stack(time_opto_2sec), axis=0)

for file_path_trial in matching_files_2sec_alltrials:
    data_trial = pd.read_csv(file_path_trial)
    nonopto_2sec += sum(data_trial.kind==2)
    opto_2sec += sum(data_trial.kind==5)
#%%
plt.plot(batch_time_nonopto_2sec, batch_mean_nonopto_2sec, color='b')
plt.fill_between(batch_time_nonopto_2sec, batch_mean_nonopto_2sec-batch_err_nonopto_2sec, batch_mean_nonopto_2sec+batch_err_nonopto_2sec, alpha=.1, color='k', lw=0)
plt.plot(batch_time_opto_2sec, batch_mean_opto_2sec, color='r')
plt.fill_between(batch_time_opto_2sec, batch_mean_opto_2sec-batch_err_opto_2sec, batch_mean_opto_2sec+batch_err_opto_2sec, alpha=.1, color='k', lw=0)
blue_patch = mpatches.Patch(color='b', label=f'non-OPTO({nonopto_2sec})')
red_patch = mpatches.Patch(color='r', label=f'OPTO({opto_2sec})')
plt.legend(handles=[blue_patch, red_patch], loc='upper left')

n,x = plt.ylim()
plt.vlines(2000, n, x, linestyle='--', color='grey')
plt.vlines(2250., n, x, linestyle='--', color='grey')

plt.xlabel('Time from CS (msec)')
plt.ylabel('Eyelid (a.u.)')
plt.savefig(f'/Users/thussenthanwalter-angelo/Desktop/Thesis/Batch Compile/Two Second Batch Compile.png')
plt.show()

plt.plot(batch_time_nonopto_2sec, batch_mean_nonopto_2sec, color='b')
plt.fill_between(batch_time_nonopto_2sec, batch_mean_nonopto_2sec-batch_err_nonopto_2sec, batch_mean_nonopto_2sec+batch_err_nonopto_2sec, alpha=.1, color='k', lw=0)
plt.plot(batch_time_opto_2sec, batch_mean_opto_2sec, color='r')
plt.fill_between(batch_time_opto_2sec, batch_mean_opto_2sec-batch_err_opto_2sec, batch_mean_opto_2sec+batch_err_opto_2sec, alpha=.1, color='k', lw=0)
blue_patch = mpatches.Patch(color='b', label=f'non-OPTO({nonopto_2sec})')
red_patch = mpatches.Patch(color='r', label=f'OPTO({opto_2sec})')
plt.legend(handles=[blue_patch, red_patch], loc='upper left')

n,x = plt.ylim()
plt.vlines(2000, n, x, linestyle='--', color='grey')
plt.vlines(2250., n, x, linestyle='--', color='grey')

plt.xlabel('Time from CS (msec)')
plt.ylabel('Eyelid (a.u.)')
plt.xlim(1900, 2500)
plt.savefig(f'/Users/thussenthanwalter-angelo/Desktop/Thesis/Batch Compile/Two Second Batch Compile_zoomed.png')
plt.show()
#%%
np.savez('/Users/thussenthanwalter-angelo/Desktop/Thesis/Batch Compile/batch_plot_2sec_processed_data{}.npy', 
         batch_mean_nonopto_2sec=batch_mean_nonopto_2sec, 
         batch_err_nonopto_2sec=batch_err_nonopto_2sec, 
         batch_time_nonopto_2sec=batch_time_nonopto_2sec, 
         batch_mean_opto_2sec=batch_mean_opto_2sec, 
         batch_err_opto_2sec=batch_err_opto_2sec, 
         batch_time_opto_2sec=batch_time_opto_2sec)
# %%
if len(batch_mean_nonopto_1sec) != len(batch_mean_opto_1sec):
    raise ValueError("Arrays must be of the same length")

t_stat, p_values = ttest_ind_from_stats(batch_mean_nonopto_1sec, batch_err_nonopto_1sec, nonopto_1sec, batch_mean_opto_1sec, batch_err_opto_1sec, opto_1sec)

p_adjusted = multipletests(p_values, method='bonferroni')
p_adjusted = p_adjusted[1]

plt.plot(batch_time_nonopto_1sec, batch_mean_nonopto_1sec, color='b')
plt.fill_between(batch_time_nonopto_1sec, batch_mean_nonopto_1sec-batch_err_nonopto_1sec, batch_mean_nonopto_1sec+batch_err_nonopto_1sec, alpha=.1, color='k', lw=0)
plt.plot(batch_time_opto_1sec, batch_mean_opto_1sec, color='r')
plt.fill_between(batch_time_opto_1sec, batch_mean_opto_1sec-batch_err_opto_1sec, batch_mean_opto_1sec+batch_err_opto_1sec, alpha=.1, color='k', lw=0)
blue_patch = mpatches.Patch(color='b', label=f'non-OPTO ({nonopto_1sec})')
red_patch = mpatches.Patch(color='r', label=f'OPTO ({opto_1sec})')
plt.legend(handles=[blue_patch, red_patch], loc='upper left')

mean_time = (batch_time_nonopto_1sec + batch_time_opto_1sec) / 2
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

plt.title('Significance Comparison of One Second Opto Stim and Non-Opto Trials')
plt.xlabel('Time from CS (msec)')
plt.ylabel('Eyelid (a.u.)')
plt.savefig(f'/Users/thussenthanwalter-angelo/Desktop/Thesis/Batch Compile/One Second Batch Compile_significance.png')
plt.show()
# %%
if len(batch_mean_nonopto_2sec) != len(batch_mean_opto_2sec):
    raise ValueError("Arrays must be of the same length")

t_stat, p_values = ttest_ind_from_stats(batch_mean_nonopto_2sec, batch_err_nonopto_2sec, nonopto_2sec, batch_mean_opto_2sec, batch_err_opto_2sec, opto_2sec)

p_adjusted = multipletests(p_values, method='bonferroni')
p_adjusted = p_adjusted[1]

plt.plot(batch_time_nonopto_2sec, batch_mean_nonopto_2sec, color='b')
plt.fill_between(batch_time_nonopto_2sec, batch_mean_nonopto_2sec-batch_err_nonopto_2sec, batch_mean_nonopto_2sec+batch_err_nonopto_2sec, alpha=.1, color='k', lw=0)
plt.plot(batch_time_opto_2sec, batch_mean_opto_2sec, color='r')
plt.fill_between(batch_time_opto_2sec, batch_mean_opto_2sec-batch_err_opto_2sec, batch_mean_opto_2sec+batch_err_opto_2sec, alpha=.1, color='k', lw=0)
blue_patch = mpatches.Patch(color='b', label=f'non-OPTO ({nonopto_2sec})')
red_patch = mpatches.Patch(color='r', label=f'OPTO ({opto_2sec})')
plt.legend(handles=[blue_patch, red_patch], loc='upper left')

mean_time = (batch_time_nonopto_2sec + batch_time_opto_2sec) / 2
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
    # Significance level alpha = 0.05 for '*' with Bonferroni Correction
    if p_adjusted[i] < 0.05:
        plt.annotate('*', (mean_time[i], y_95_percent), textcoords="offset points", xytext=(0, 10), ha='center', color='red')
    # Significance level alpha = 0.01 for '**' with Bonferroni Correction
    if p_adjusted[i] < 0.01:
        plt.annotate('*', (mean_time[i], y_95_percent), textcoords="offset points", xytext=(0, 5), ha='center', color='red')
    # Significance level alpha = 0.001 for '***' with Bonferroni Correction
    if p_adjusted[i] < 0.001:
        plt.annotate('*', (mean_time[i], y_95_percent), textcoords="offset points", xytext=(0, 0), ha='center', color='red')
        
n,x = plt.ylim()
plt.vlines(2000, n, x, linestyle='--', color='grey')
plt.vlines(2250., n, x, linestyle='--', color='grey')
plt.xlim(1900, 2500)

plt.title('Significance Comparison of Two Second Opto Stim and Non-Opto Trials')
plt.xlabel('Time from CS (msec)')
plt.ylabel('Eyelid (a.u.)')
plt.savefig(f'/Users/thussenthanwalter-angelo/Desktop/Thesis/Batch Compile/Two Second Batch Compile_significance.png')
plt.show()
# %%
