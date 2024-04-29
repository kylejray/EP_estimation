from mpi4py import MPI
comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

import sys, os
sys.path.append(os.path.expanduser('~/source/'))
from kyle_tools import save_as_json

import numpy as np
from weight_discovery_oneshot import *
from free_diffusion import *
from oneshot_running_parameters import run_params as rp

from time import sleep

init_params = rp['init_params']
params['init'] = init_params

#kwargs = {'cyclic_lims':None, 'padding':-2}

T = rp['T'] # Total time
dt = rp['dt']
num_steps = rp['num_steps']
skip = rp['skip']

calc_params = {}

num_paths_list = rp['num_paths_list']
num_ests = rp['num_ests']

num_iter = rp['num_iter']
max_size = rp['max_size']


batch_size = int(num_iter/size)

for num_paths in num_paths_list:
    sim_params = [num_paths, num_steps, dt]
    proc_params = [num_paths, int(num_steps/skip), skip*dt]

    params['sim'] = sim_params

    calc_params.update(params)
    calc_params['sim'] = proc_params

    for num_est in num_ests:

        estimates=[]
        Xis = []
        mus = []
        print(f'rank {rank} run starting {num_est} estimators')
        sys.stdout.flush()
        sleep(10*max_size*rank)
        for i in range(batch_size):
            process = simulate_free_diffusion_underdamped(params, save_skip=skip)

            print(f'rank {rank} got process for run {i} ')
            sys.stdout.flush()
            
            if max_size is not None:
                total_gbytes = np.prod(process.shape[:-1])*np.prod(num_est)*8/(10**9)
                n_chunks = ceil(total_gbytes / max_size)
                max_steps = ceil(process.shape[0]/n_chunks)
                k_params = get_kernel_params(process, num_est)
                
                mu = np.zeros(np.prod(num_est))
                Xi = np.zeros((np.prod(num_est),np.prod(num_est)))
                j=0
                while j < process.shape[0]:
                    print(f'rank {rank} calculating trials [{j}:{j+max_steps}]')
                    sys.stdout.flush()
                    proc = process[j:j+max_steps]
                    mu_temp, Xi_temp = get_ep(proc, calc_params, num_est, return_mats=True, kernel_params=k_params)
                    mu += mu_temp*proc.shape[0]/process.shape[0]
                    Xi += Xi_temp*proc.shape[0]/process.shape[0]
                    j += max_steps
            

            else:
                mu, Xi = get_ep(process, calc_params, num_est, return_mats=True)

            print(f'rank {rank} got matrices for run {i} ')
            sys.stdout.flush()

            estimates.append(mu@np.linalg.inv(Xi)@mu)
            Xis.append(Xi)
            mus.append(mu)
            
            print(f'rank{rank} finished {i} run of {num_est} estimators')
            sys.stdout.flush()

        print(f'rank {rank} waiting')
        sys.stdout.flush()
        comm.Barrier()

        d = np.array(comm.gather(estimates))
        x = np.array(comm.gather(Xis))
        m = np.array(comm.gather(mus))

        if rank == 0:
            print('gathering done')
            sys.stdout.flush()
            da, xa, ma = [a.reshape(a.shape[0]*a.shape[1],*a.shape[2:]) for a in [d,x,m]]

            output = {'data':da}
            output['Xis'] = xa
            output['mus'] = ma
            output['params'] = params
            output['proc_params'] = calc_params

            output['num_est'] = num_est
            t = np.arange(0,T, dt)
            output['theo'] = np.sum(realepr(t,*params['init']))*dt

            save_as_json(output, name=f'N{int(num_paths/1_000):03}_E{num_est}_steps{int(num_steps/skip):04}')