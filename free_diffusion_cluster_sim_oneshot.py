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


init_params = [.2,.3,.025]
params['init'] = init_params

T = .25# Total time
dt = 0.0025
num_steps = int(T / dt)
skip=1
calc_params = {}


num_paths_list = [25_000]
num_ests = [[6, 6, 6], [6,6,8], [8,6,6], [6,8,6]]


num_iter = 5
batch_size = int(num_iter/size)

for num_paths in num_paths_list:
    sim_params = [num_paths, num_steps, dt]
    params['sim'] = sim_params

    calc_params.update(params)
    calc_params['sim'][2] = dt*skip

    for num_est in num_ests:

        estimates=[]
        Xis = []
        mus = []
        print(f'rank {rank} run starting {num_est} estimators')
        for i in range(batch_size):
            process = simulate_free_diffusion_underdamped(params, save_skip=skip)
            
            mu, Xi = get_ep(process, calc_params, num_est, return_mats=True)
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