from mpi4py import MPI
comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

import sys, os
sys.path.append(os.path.expanduser('~/source/'))
from kyle_tools import save_as_json

import numpy as np
from weight_discovery import *

#import process data
files = [f'./data/jinghao_mystery_traj_dt_p001_ep_p25_{n}.npy' for n in [f'0{N+1}' for N in range(5,9)]]
process = ([np.array(np.load(item)).squeeze()[:,:,:] for item in files])
process = np.vstack(process)
np.random.shuffle(process)

d=2
s =  process.shape
new_len = int(s[1]/d)
new_process = np.empty( (d*s[0], new_len, s[-1]) )
for i in range(d):
    new_process[i*s[0]:(i+1)*s[0],...] = process[:,i*new_len:(i+1)*new_len ,...]
#new_process = new_process.reshape(d*s[0]*s[1], new_len,s[-1])
print(process.shape, new_process.shape)

process = new_process
process = process[:,:10,:]

#set parameters to test
gamma = 1
proc_dt = .001
kwargs = {'cyclic_lims':[-1.5,1.5]}

steps = range(process.shape[1]-2)

num_paths = [500_000, 1_000_000, 2_400_000, 4_800_000]
num_ests = [6, 8, 10, 12, 14, 16]

NE = np.meshgrid(num_paths, num_ests)
param_list = np.vstack([ i.ravel() for i in NE]).T

rank_list = [ param_list[i::size] for i in range(size)]

#distribute to diff tasks
params = rank_list[rank]
for N,E in params:
    proc = process[:N]
    eps=[]
    for time_step in steps:
        
        eps.append(get_step_epr(proc, time_step, proc_dt, gamma, E, **kwargs ))
    
    print(f'rank{rank}: N={N/1E6:.2f}e6 Num_Est={E}. Avg {np.mean(eps):.2f}')
    sys.stdout.flush()
    temp = {'num_path':N, 'num_est':E, 'dt':proc_dt, 'gamma':gamma, 'data':eps}
    save_as_json(temp, name=f'N{N}E{E}')