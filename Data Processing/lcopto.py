# -*- coding: utf-8 -*-
"""
Created on Wed Jun 28 13:59:02 2023

@author: wanglab
"""

## Two CSs codes. 

#%% import libraries
%matplotlib auto
import os,h5py, numpy as np, pandas as pd, cv2
import matplotlib.pyplot as pl

#%% determine data file names

# path to directory containing all data
#path = '/home/wanglab/wang/Junuk/eyeblink/lc_axon_dcn_202105/20210605_ebc28/g10'
path =   '/home/wanglab/wang/Junuk/eyeblink/2023_ebc_pcp_ai27/20230714_ebc16/lcop01/'

#path = '/jukebox/wang/Junuk/eyeblink/CNO_1/20190809/i4'

# collect h5 files:
files = sorted([(d,f) for d,_,fs in os.walk(path) for f in fs if f.endswith('.h5')])
# infer csv files from h5 files:
csv_files = [os.path.join(d, os.path.splitext(f)[0].replace('video','data')+'.csv') for d,f in files]
h5_files = [os.path.join(*i) for i in files]
names = [os.path.splitext(os.path.split(f)[-1])[0].replace('_video','') for f in h5_files]
# confirm that files all exist
for c,h in zip(csv_files, h5_files):
    if not os.path.exists(c):
        print('Failed to find {}'.format(c))
    if not os.path.exists(h):
        print('Failed to find {}'.format(h))

#%% select dataset to analyze

# show available data and select one to analyze
print('Available datasets:')
for idx,n in enumerate(names):
    print('\t{}\t{}'.format(idx,n))

# select file index based on available datasets
file_idx = 0
mov_idx = 0 #0 or 1

csv = csv_files[file_idx]
h5 = h5_files[file_idx]
ddir,_ = os.path.split(h5)
name = names[file_idx]

print('\nSelection:\tindex', file_idx)
print('\t\t'+name)
print('\t\t'+csv)
print('\t\t'+h5)


#%% read in some initial data

# load csv (with trial timing info)
data = pd.read_csv(csv)
# load example frames
movname = 'mov{}'.format(mov_idx) # movie frames
tsname = 'ts{}'.format(mov_idx) # timestamps
with h5py.File(h5, 'r') as h:
    nsamp = len(h[movname])
    print(nsamp, 'frames in dataset.')
    #mov = np.array(h[movname][1000:1050]) / 255
    mov = np.array(h[movname][26000:26050]) / 255
    
    ts = np.array(h[tsname])
    ts = ts[:,0]

#%% select ROI for eye analysis (skip if not 1st time)

# select eye ROI
pl.imshow(mov.mean(axis=0), cmap=pl.cm.Greys_r)
pts = pl.ginput(timeout=-1, n=-1)
pl.close()

# convert points to mask
pts = np.asarray(pts, dtype=np.int32)
roi = np.zeros(mov[0].shape, dtype=np.int32)
roi = cv2.fillConvexPoly(roi, pts, (1,1,1), lineType=cv2.LINE_AA)
roi = roi.astype(np.float)

# save selected ROI to data directory
roiname = os.path.join(ddir, name+'_roi{}.npy'.format(mov_idx))
np.save(roiname, roi)


#%% extract trace from entire movie (takes some time) (skip if not 1st time)
trace = np.empty(nsamp)
chunk_size = 5000
niter = int(np.ceil(nsamp/chunk_size))
for idx in range(niter):
    i0,i1 = idx*chunk_size, idx*chunk_size+chunk_size
    if i1 > nsamp:
        i1 = nsamp
    print ('{}-{} / {}'.format(i0,i1,nsamp))
    with h5py.File(h5, 'r') as h:
        mov = np.array(h[movname][i0:i1]) / 255

    tr = mov.reshape([len(mov), -1]) @ roi.reshape(np.product(roi.shape))
    tr /= np.sum(roi)
    trace[i0:i1] = tr

# save extracted trace to data directory
trname = os.path.join(ddir, name+'_trace{}.npy'.format(mov_idx))
np.save(trname, trace)


#%% load the extracted trace
trname = os.path.join(ddir, name+'_trace{}.npy'.format(mov_idx))
trace = np.load(trname)

#%% make a simple plot of mean intensity in eye ROI over CSUS trials

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

Ts = np.mean(np.diff(ts))

csus = data[data.kind==2]# 0:CS_ONLY, #1: US_ONLY, 2:CSUS s
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
plt.legend(handles=[red_patch], loc='upper left')

n,x = pl.ylim()
pl.vlines(1000, n, x, linestyle='--', color='grey')
pl.vlines(1250., n, x, linestyle='--', color='grey')


pl.xlabel('Time from CS (msec)')
pl.ylabel('Eyelid (a.u.)')

plot_filename = os.path.join(ddir, name+'_plot{}.jpg'.format(mov_idx))
pl.savefig(plot_filename)

proc_filename = os.path.join(ddir, name+'_processed_data{}.npy'.format(mov_idx))
np.savez(proc_filename, mean=mean, std=err, time=time)

#np.reshape(mean, (250,1))#%%save slices as excel file

#%%
import pandas as pd

## convert your array into a dataframe
df = pd.DataFrame (slices)

## save to xlsx file
filepath = '/home/wanglab/wang/Junuk/eyeblink/2023_ebc_pcp_ai27/20230714_ebc16/lcop01/nonopto_20230714_lcop01.csv'
#filepath = '/home/wanglab/wang/Junuk/eyeblink/lc_axon_dcn_202105/20210605_ebc28/g10/CS_US_0605_g10.csv'

df.to_csv(filepath, index=False)

