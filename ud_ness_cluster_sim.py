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
data_range = range(5,13)

files = [f'./NESS_data/jinghao_mystery_traj_dt_p001_ep_p25_{n}.npy' for n in [f'{N:02}' for N in data_range]]
process = ([np.array(np.load(item)).squeeze()[:,:,:] for item in files])
process = np.vstack(process)

d=3
s =  process.shape
new_len = int(s[1]/d)
new_process = np.empty( (d*s[0], new_len, s[-1]) )
for i in range(d):
    new_process[i*s[0]:(i+1)*s[0],...] = process[:,i*new_len:(i+1)*new_len ,...]
#new_process = new_process.reshape(d*s[0]*s[1], new_len,s[-1])
if rank == 0:
    print(process.shape, new_process.shape)
    sys.stdout.flush()


process = new_process


randomize_steps = False
shuffle_trials = True

if shuffle_trials:
    np.random.shuffle(process)

#set parameters to test
gamma = 1
proc_dt = .001

kwargs = {'cyclic_lims':[-1.5,1.5], 'padding':-2}

num_paths = [10_800_000]

num_ests = [21]

NE = np.meshgrid(num_paths, num_ests)

param_list = list(np.vstack([ i.ravel() for i in NE]).T)


rank_list = [ param_list[i::size] for i in range(size)]

if rank==0:
    print(f'steps are randomized:{randomize_steps}')
    print(f'trials are shuffled:{shuffle_trials}')
    print(f'using data {data_range}')
    sys.stdout.flush()

process = process[:max(num_paths)]
steps = range(process.shape[1]-2)
if rank ==0:
    print(steps)

#distribute to diff tasks
params = rank_list[rank]
#choose random or sequential paths

#ovveride
#params = [[500_000, 15]]

print(f'rank {rank} param_list:{params}')
sys.stdout.flush()

for N,E in params:
    eps = []
    mus = []
    Xis = []

    #proc = process[:N]
    

    steps_idx = np.meshgrid(steps, np.empty(N))[0]
    if randomize_steps:
        steps_idx = np.random.default_rng().permuted(steps_idx,axis=1)

    counter = 0
    ratio = 0
    for time_step in steps:
        counter += 1
        if ratio < float(f'{counter/len(steps):.1f}'):
            ratio = float(f'{counter/len(steps):.1f}')
            print(f'rank{rank} with N={N} ad E={E} done with {ratio} proportion of steps')
            sys.stdout.flush()


        proc = np.empty( (N,3,2) )

        for j in range(3):
            proc[:,j] = process[range(N),steps_idx[:,0]+j]
        steps_idx = steps_idx[:,1:]

        mu, Xi = get_step_epr(proc, 0, proc_dt, gamma, E, return_mats=True, **kwargs )
        eps.append((mu@np.linalg.inv(Xi)@mu)/proc_dt)
        mus.append(mu)
        Xis.append(Xi)

        
    
    print(f'rank{rank}: N={N/1E6:.2f}e6 Num_Est={E}. Avg {np.mean(eps):.3f}')
    sys.stdout.flush()
    temp = {'num_path':N, 'num_est':E, 'dt':proc_dt, 'gamma':gamma, 'data':eps, 'data_range':list(data_range), 'rand_steps':randomize_steps, 'shuffle_trials':shuffle_trials}
    save_as_json(temp, name=f'shuffle_N{N:08}E{E:02}')
    temp.update({'mus':mus, 'Xis':Xis})
    save_as_json(temp, name=f'shuffle_N{N:08}E{E:02}_mats')



print(f'rank{rank} done')
sys.stdout.flush()