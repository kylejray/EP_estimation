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
def get_kernel_params(process, time_step, num_estimators, padding=2, **kwargs):
    lowers, uppers = one_step_stats(process, time_step)
    # The region we want to cover [x_lower, x_upper]*[p_lower, p_upper]
    std_devs = (uppers-lowers)/(num_estimators-padding)
    means = np.linspace(lowers-std_devs/padding, uppers+std_devs/padding, num_estimators)
    return means, std_devs

#return the an array that has all basis func evaluated at all data points at the time step
# shape will be [num_paths, num_estimators, num_estimators]
def gaussian_basis_function(process, time_step, num_estimators, kernel_params=None, **kwargs):
    if kernel_params is None:
        means, std_devs = get_kernel_params(process, time_step, num_estimators, **kwargs)
    else:
        means, std_devs = kernel_params

    mean_x, mean_p = np.meshgrid(*means.T)
    std_dev_x, std_dev_p = std_devs

    position = process[:,time_step,0][:,None,None]
    momentum = process[:,time_step,1][:,None,None]
    
    kernel_value = np.exp(-(momentum - mean_p)**2/(2 * std_dev_p**2)-(position - mean_x)**2/(2 * std_dev_x**2))
    return(kernel_value)


def gaussian_periodic_basis_function(process, time_step, num_estimators, kernel_params=None, cyclic_lims=None, **kwargs):
    if kernel_params is None:
        means, std_devs = get_kernel_params(process, time_step, num_estimators, **kwargs)
    else:
        means, std_devs = kernel_params
    
    means = means.T

    if cyclic_lims:
        domain = np.diff(np.asarray(cyclic_lims))
        num_periodic = int(np.ceil((num_estimators/2)))
        means = [np.linspace(1,num_periodic,num_periodic), means[1,:]]
        kernel_value = np.empty((process.shape[0],2*num_periodic, num_estimators))
        

    mean_x, mean_p = np.meshgrid(*means)
    std_dev_x, std_dev_p = std_devs
    
    position = process[:,time_step,0][:,None,None]
    momentum = process[:,time_step,1][:,None,None]

    if cyclic_lims:
        kernel_value[...,:num_periodic] = np.exp(-(momentum - mean_p)**2/(2 * std_dev_p**2))*(.501+.5*np.sin(mean_x*position*2*np.pi/domain))
        kernel_value[...,:num_periodic] = np.exp(-(momentum - mean_p)**2/(2 * std_dev_p**2))*(.501+.5*np.cos(mean_x*position*2*np.pi/domain))
        return kernel_value

    kernel_value = np.exp(-(momentum - mean_p)**2/(2 * std_dev_p**2)-(position - mean_x)**2/(2 * std_dev_x**2))
    return(kernel_value)

#
def gaussian_basis_mu_vector(process, index, num_estimators, gamma, kernels=None, cyclic_lims=None, **kwargs):
    
    # Initialize the current value
    current = 0

    # Iterate for each time step and each path
    for time in range(index, index+1):
        if kernels is None:
            w = kernels[0]
            dw = kernels[0] - w
        # Compute dx, dw, w and dp for the step for all paths at once
        else:
            w = gaussian_basis_function(process, time, num_estimators, **kwargs)
            dw = gaussian_basis_function(process, time+1, num_estimators, **kwargs)- w

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
    return current_average.reshape(-1)

#
def gaussian_basis_Xi_matrix(process, index, num_estimators, dt, kernel_value=None, **kwargs):
    if kernel_value is None:
        kernel_value = gaussian_basis_function(process, index, num_estimators, **kwargs)
    # Initialize Xi matrix
    Xi_matrix = np.zeros((num_estimators**2,num_estimators**2))

    # turn the square basis matrix into a vector, then take innver product
    # could be optimized since we know its symmetric, but I didnt bother
    N = process.shape[0]

    basis_vector = kernel_value.reshape(kernel_value.shape[0],-1)

    Xi_matrix += dt * np.einsum('ij,ik->jk', basis_vector, basis_vector)

    # average over paths
    return Xi_matrix / N


def get_step_epr_periodic(process, index, dt, gamma, num_est, return_mats = False, kernel_params=None, **kwargs):

    kernel = gaussian_periodic_basis_function(process, index, num_est, kernel_params= kernel_params, **kwargs)
    next_kernel = gaussian_periodic_basis_function(process, index+1, num_est, kernel_params= kernel_params, **kwargs)

    mu = gaussian_basis_mu_vector(process, index, num_est, gamma, kernels=[kernel, next_kernel], **kwargs)
    Xi = gaussian_basis_Xi_matrix(process, index, num_est, dt, kernel_value=kernel, **kwargs)
    
    if return_mats:
        return mu, Xi
    
    ep = mu@np.linalg.inv(Xi)@mu
    return ep/dt


def get_step_epr(process, index, dt, gamma, num_est, return_mats = False, kernel_params=None, **kwargs):

    kernel = gaussian_basis_function(process, index, num_est, kernel_params= kernel_params, **kwargs)
    next_kernel = gaussian_basis_function(process, index+1, num_est, kernel_params= kernel_params, **kwargs)

    mu = gaussian_basis_mu_vector(process, index, num_est, gamma, kernels=[kernel, next_kernel], **kwargs)
    Xi = gaussian_basis_Xi_matrix(process, index, num_est, dt, kernel_value=kernel, **kwargs)
    
    if return_mats:
        return mu, Xi
    
    ep = mu@np.linalg.inv(Xi)@mu
    return ep/dt

def add_regularizer(data_dict, dt, alpha_gen = .01, verbose=False):
    output = []
    reg_Xis = []
    for item,m in zip(data_dict['Xis'], data_dict['mus']):

        try:        alpha = alpha_gen(item)
        except:     alpha = alpha_gen

        X = item + alpha*np.identity( len(item))

        reg_Xis.append(X)
        output.append(m @ np.linalg.inv(X) @ m / dt)

    data_dict['reg_Xis'] = reg_Xis
    data_dict['reg_data'] = output
    if verbose:
        om, nm = np.mean(data_dict['data']), np.mean(output)
        os, ns = 3*np.std(data_dict['data'])/np.sqrt(len(output)), 3*np.std(output)/np.sqrt(len(output))
        print(f'mean, var changed from {om:.4f},{os:.6f} to {nm:.4f},{ns:.6f}')
    return
    

    



