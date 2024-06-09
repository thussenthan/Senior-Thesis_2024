#%% import libraries
import matplotlib.pyplot as pl
import matplotlib.patches as mpatches
import os,h5py, numpy as np, pandas as pd, cv2

#%%
path =   '/Volumes/wang/Junuk/eyeblink/2023_ebc_pcp_ai27/20230720_ebc21/lcop02light'

files = sorted([(d,f) for d,_,fs in os.walk(path) for f in fs if f.endswith('.h5')])

csv_files = [os.path.join(d, os.path.splitext(f)[0].replace('video','data')+'.csv') for d,f in files]
h5_files = [os.path.join(*i) for i in files]
names = [os.path.splitext(os.path.split(f)[-1])[0].replace('_video','') for f in h5_files]

file_idx = 0
mov_idx = 0

csv = csv_files[file_idx]
h5 = h5_files[file_idx]
ddir,_ = os.path.split(h5)
name = names[file_idx]

data = pd.read_csv(csv)
tsname = 'ts{}'.format(mov_idx) # timestamps
with h5py.File(h5, 'r') as h:    
    ts = np.array(h[tsname])
    ts = ts[:,0]

#%%
# load the extracted trace
trname = os.path.join(ddir, name+'_trace{}.npy'.format(mov_idx))
trace = np.load(trname)

#%% REGEN NON-OPTO
# make a simple plot of mean intensity in eye ROI over CSUS trials

Ts = np.mean(np.diff(ts))

csus = data[data.kind==2]# 0:CS_ONLY, #1: US_ONLY, 2:CSUS
cstimes = csus.cs_ts0.values
# ustimes = csus.us_ts0.values

cs_frames = np.array([np.argmin(np.abs(c-ts)) for c in cstimes])
#cs_frames = np.array([np.argmin(np.abs(c-ts)) for c in ustimes])


pad = (60, 500) #(60, 191) 
slices = np.array([trace[f-pad[0]:f+pad[1]] for f in cs_frames])

time = np.arange(-pad[0], pad[1]) * Ts
time *= 1000 #1000
err = slices.std(axis=0)
mean = slices.mean(axis=0)

pl.plot(time, mean, color='b')
pl.fill_between(time, mean-err, mean+err, alpha=.1, color='k', lw=0)
red_patch = mpatches.Patch(color='b', label='non-OPTO')
pl.legend(handles=[red_patch], loc='upper left')

n,x = pl.ylim()
pl.vlines(1000, n, x, linestyle='--', color='grey')
pl.vlines(1250., n, x, linestyle='--', color='grey')

pl.xlabel('Time from CS (msec)')
pl.ylabel('Eyelid (a.u.)')

plot_filename = os.path.join(ddir, name+'_plot{}.jpg'.format(mov_idx))
pl.savefig(plot_filename)

proc_filename = os.path.join(ddir, name+'_nonopto_processed_data{}.npy'.format(mov_idx))
np.savez(proc_filename, mean=mean, std=err, time=time)

#np.reshape(mean, (250,1))#%%save slices as excel file

#%% REGEN OPTO
#make a simple plot of mean intensity in eye ROI over CSUS trials

Ts = np.mean(np.diff(ts))

csus = data[data.kind==5]# 0:CS_ONLY, #1: US_ONLY, 2:CSUS s
cstimes = csus.cs_ts0.values
# ustimes = csus.us_ts0.valueslcop01

cs_frames = np.array([np.argmin(np.abs(c-ts)) for c in cstimes])
#cs_frames = np.array([np.argmin(np.abs(c-ts)) for c in ustimes])

pad = (60, 500) #(60, 191) 
slices = np.array([trace[f-pad[0]:f+pad[1]] for f in cs_frames])

time = np.arange(-pad[0], pad[1]) * Ts
time *= 1000 #1000
err = slices.std(axis=0)
mean = slices.mean(axis=0)

pl.plot(time, mean, color='r')
pl.fill_between(time, mean-err, mean+err, alpha=.1, color='k', lw=0)
red_patch = mpatches.Patch(color='r', label='OPTO')
pl.legend(handles=[red_patch], loc='upper left')

n,x = pl.ylim()
pl.vlines(1000, n, x, linestyle='--', color='grey')
pl.vlines(1250., n, x, linestyle='--', color='grey')


pl.xlabel('Time from CS (msec)')
pl.ylabel('Eyelid (a.u.)')

plot_filename = os.path.join(ddir, name+'_plot{}.jpg'.format(mov_idx))
pl.savefig(plot_filename)

proc_filename = os.path.join(ddir, name+'_opto_processed_data{}.npy'.format(mov_idx))
np.savez(proc_filename, mean=mean, std=err, time=time)

#np.reshape(mean, (250,1))#%%save slices as excel file