from mpi4py import MPI
comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

import sys, os
sys.path.append(os.path.expanduser('~/source/'))
from kyle_tools import save_as_json

import numpy as np
from weight_discovery import *
from free_diffusion import *


init_params = [.2,.3,.025]
params['init'] = init_params

T = .25# Total time
dt = 0.001
num_steps = int(T / dt)
num_paths = 10_000

sim_params = [num_paths, num_steps, dt]
params['sim'] = sim_params

skip=1
kwargs = {'cyclic_lims':[-1.5,1.5], 'padding':2}

num_iter = 30
num_ests = [6, 8, 10,]

#steps = [ int(5*i) for i in  [1, 2, 4.2, 10, 15, 20, 25, 35, 47]]
#steps = [ int(i) for i in np.linspace(1, num_steps-2, num_steps-2)]
steps = list(range(int(num_steps/skip)-2))


batch_size = max(int(num_iter/size),1)
if rank==0:
    print(f'N={num_paths}, E={num_ests}, iter={num_iter}, batch_size={batch_size}')
    sys.stdout.flush()

all_estimates = []
for i in range(batch_size):

    process = simulate_free_diffusion_underdamped(params, save_skip=skip)
    print(f' rank{rank} sim {i} done', end='\r')
    sys.stdout.flush()
    estimates = []
    for num_est in num_ests:
        n_resolved_estimates=[]
        for time_step in steps:
            ep = get_step_epr(process, time_step, dt, params['gamma'], num_est, **kwargs)
            n_resolved_estimates.append(ep)
        estimates.append(n_resolved_estimates)
        print(f'rank{rank} finished {i} run of {num_est} estimates', end='\r')
        sys.stdout.flush()
    all_estimates.append(estimates)


comm.Barrier()
print('starting collection')
sys.stdout.flush()
d = np.array(comm.gather(all_estimates))

print('ending collection')
sys.stdout.flush()
if rank == 0:
    

    data = d.reshape(-1,*d.shape[2:])

    output = {'data':data}
    output['params'] = params
    output['steps'] = steps
    output['num_ests'] = num_ests
    t = np.arange(0,T, dt*skip)
    output['t'] = t
    output['theo'] = [t, realepr(t,*params['init'])] 

    save_as_json(output, name=f'N{num_paths}_steps{len(steps)}')