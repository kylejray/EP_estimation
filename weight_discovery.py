import numpy as np

def data_t_tplus(process, time_step):
    datat = process[:,[time_step,time_step+1],:]
    return datat

# get measurements about a time step
def one_step_stats(process, time_step):
    data = data_t_tplus(process, time_step)
    maxs = np.max(data, axis=(0,1))
    mins = np.min(data, axis=(0,1))
    stds = np.std(data[:,0,:], axis=(0))
    lowers, uppers = mins - stds, maxs + stds
    #lowers, uppers = mins, maxs

    return [lowers, uppers]

# turn the measurements into bin centers and std dev
def get_kernel_params(process, time_step, num_estimators, padding=2):

    lowers, uppers = one_step_stats(process, time_step)
    # The region we want to cover [x_lower, x_upper]*[p_lower, p_upper]
    std_devs = (uppers-lowers)/(num_estimators-padding)
    means = np.linspace(lowers-std_devs/padding, uppers+std_devs/padding, num_estimators)
    return means, std_devs

#return the an array that has all basis func evaluated at all data points at the time step
# shape will be [num_paths, num_estimators, num_estimators]
def gaussian_basis_function(process, time_step, num_estimators):
    means, std_devs = get_kernel_params(process, time_step, num_estimators)

    mean_x, mean_p = np.meshgrid(*means.T)
    std_dev_x, std_dev_p = std_devs

    position = process[:,time_step,0][:,None,None]
    momentum = process[:,time_step,1][:,None,None]
    
    kernel_value = np.exp(-(momentum - mean_p)**2/(2 * std_dev_p**2)-(position - mean_x)**2/(2 * std_dev_x**2))
    return(kernel_value)

#
def gaussian_basis_irr_current_average(process, index, num_estimators, gamma, cyclic_lims=None):
    # Initialize the current value
    current = 0

    # Iterate for each time step and each path
    for time in range(index, index+1):
        # Compute dx, dw, w and dp for the step for all paths at once
        w = gaussian_basis_function(process, time, num_estimators)
        dw = gaussian_basis_function(process, time+1, num_estimators)- w

        step_diff = np.diff(data_t_tplus(process, time), axis=1).squeeze()


        dx, dp = step_diff[...,0], step_diff[...,1]
        if cyclic_lims is not None:
            domain = np.diff(np.asarray(cyclic_lims))
            jump_back = dx < -.8*domain
            jump_forw = dx > .8*domain
            dx[jump_back] = dx[jump_back] + domain
            dx[jump_forw] = dx[jump_forw] - domain

        # The expression of our current J
        current += gamma *  w * dx[:,None,None] - 1/2 * dw * dp[:,None,None]
    #average over num_paths
    current_average = current.mean(axis=0)
    return current_average

#
def gaussian_basis_Xi_matrix(process, index, num_estimators, dt):
    # Initialize Xi matrix
    Xi_matrix = np.zeros((num_estimators**2,num_estimators**2))

    # turn the square basis matrix into a vector, then take innver product
    # could be optimized since we know its symmetric, but I didnt bother
    N = process.shape[0]

    basis_vector = gaussian_basis_function(process, index, num_estimators)
    basis_vector = basis_vector.reshape(basis_vector.shape[0],-1)

    Xi_matrix += dt * np.einsum('ij,ik->jk', basis_vector, basis_vector)

    # average over paths
    return Xi_matrix / N

def gaussian_basis_mu_vector(process, index, num_estimators, gamma, **kwargs ):
    # Initialize mu vector
    mu = np.zeros(num_estimators**2)
    # all we do is reshape the average so its a vecotr instead of a square matrix
    mu += gaussian_basis_irr_current_average(process, index, num_estimators, gamma, **kwargs).reshape(-1)
    return mu


def get_step_epr(process, index, dt, gamma, num_est, return_mats = False, **kwargs):
    mu = gaussian_basis_mu_vector(process, index, num_est, gamma, **kwargs)
    Xi = gaussian_basis_Xi_matrix(process, index, num_est, dt)
    
    if return_mats:
        return mu, Xi
    
    ep = mu@np.linalg.inv(Xi)@mu
    return ep/dt

