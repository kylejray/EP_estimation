

# here are parameters that the free_diffusion_cluster_sim_oneshot.py uses
run_params={}

run_params['init_params'] = [.2,.3,.025]

run_params['kwargs'] = {'cyclic_lims':None, 'padding':-2}

T, dt = .25, .001 
run_params['T'] = T# Total time
run_params['dt'] = dt# dt for the sim
run_params['num_steps'] = int(T / dt) #nsteps for simulation

run_params['skip'] = 1 #skip_steps when saving sim output

run_params['num_paths_list'] = [50_000]
#run_params['num_paths_list'] = [10_000, 25_000, 50_000, 100_000, 250_000]

#run_params['num_ests'] = [[6,6,6], [8,6,6], [6,8,6], [8,8,6], [9,8,6], [10,8,6], [10,10,8]]
#run_params['num_ests'] = [[9,8,6], [8,8,8], [10,8,6], [10,10,8]]
run_params['num_ests'] = [[6,6,6],[6,6,8],[6,6,10]]


run_params['num_iter'] = 9 #num of iterations

run_params['max_size'] = 2 #max size in GB, footprint is ~3 times larger