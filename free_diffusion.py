import numpy as np

keys = ['sim','init','gamma','kBT', 'mass']
vals = [ [1000, 100, .01], [1,1,1], 1, 1, 1 ]

params = { k:v for k,v in zip(keys, vals)}

def simulate_free_diffusion_underdamped(all_params = params, save_skip=1):
    '''
    all_params is a dict with keys ['sim,'init','thermal'], each key is a 3 element list
        sim = [N, nun_steos, dt]
        init = p0, Sp0, Sx0
        'gamma', 'kBT', 'mass' as expected

    save_skip = n skips every 'n' sim steps when saving the output 
    '''
    N, num_steps, dt = params['sim']
    p0, Sp0, Sx0 = params['init']
    gamma, kBT, mass  = [params[k] for k in ['gamma','kBT', 'mass'] ]
    sigma = np.sqrt(2 * gamma * kBT)

    save_indices = range(0,num_steps+1)[::save_skip]
    # Initialize the array to store x position and x momentum (x,px)
    phase_data = np.zeros((N, len(save_indices), 2))
    # Sample the initial momentum 
    phase_data[:,0] = np.random.normal([0,p0],[Sx0,Sp0], (N,2))
    # initialize vars
    i=0
    curr_data = phase_data[:,i]
     # Iterate over each time step and simulate the process
    for t in range(1, num_steps + 1):
        curr_data = curr_data + dt * (curr_data[...,1][:,None] * np.array([1,-gamma]))/mass
        curr_data[:,1] += sigma * np.random.normal(0,np.sqrt(dt), N)

        if t in save_indices:
            i += 1
            phase_data[:,i] = curr_data

    return phase_data


def realepr(t, P0, SigmaPi, SigmaXi):
    '''
    gives the epr as a function of time assuming gamma=kBT=mass=1
    '''
    numerator = (np.exp(-2*t) * (np.exp(4*t) + 2*np.exp(3*t)*(-2 + SigmaPi**2) -
                                 4*np.exp(t)*(-2 + SigmaPi**2)*(-1 + P0**2 + SigmaPi**2) +
                                 (-1 + P0**2 + SigmaPi**2)*(-SigmaXi**2 - 2*(2 + t) +
                                  SigmaPi**2*(3 + SigmaXi**2 + 2*t)) +
                                 np.exp(2*t)*(7 - 7*SigmaPi**2 + SigmaPi**4 +
                                 P0**2*(-4 + SigmaPi**2 + SigmaXi**2 + 2*t))))
    denominator = (-4 - 4*np.exp(t)*(-2 + SigmaPi**2) - SigmaXi**2 - 2*t +
                   SigmaPi**2*(3 + SigmaXi**2 + 2*t) +
                   np.exp(2*t)*(-4 + SigmaPi**2 + SigmaXi**2 + 2*t))
    return numerator / denominator

