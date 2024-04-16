import numpy as np
from math import ceil

def process_stats(process):
    data = process
    maxs = np.max(data, axis=(0,1))
    mins = np.min(data, axis=(0,1))

    #stds = np.std(data[:,0,:], axis=(0))
    lowers, uppers = mins, maxs



    return [lowers, uppers]

def get_kernel_params(process, num_estimators, padding=2):
    num_estimators = np.array(num_estimators)
    lowers, uppers = process_stats(process)
    bin_widths = list((uppers-lowers)/(num_estimators[:-1]-padding))
    means = [np.linspace(l-w, u+w, n) for l,u,w,n in zip(lowers, uppers, bin_widths, num_estimators[:-1])]

    bin_widths.append(process.shape[1] / (num_estimators[-1]-padding))
    means.append(np.linspace(0-bin_widths[-1]/padding, process.shape[1]+bin_widths[-1]/padding, num_estimators[-1]))

    return [means, bin_widths]

#return the an array that has all basis func evaluated at all data points at the time step
# shape will be [num_paths, num_estimators, num_estimators]
def gaussian_basis_kernel(process, num_estimators, kernel_params=None):
    if kernel_params is None:
        means, std_devs = get_kernel_params(process, num_estimators)
    else:
        means, std_devs = kernel_params
    
    gbk = np.zeros((np.prod(num_estimators),process.shape[0], process.shape[1]))

    nPaths, nSteps, _ = process.shape

    time = np.linspace(np.zeros(nPaths), np.full(nPaths, nSteps-1), nSteps).T
    
    vals = [process[:,:,0], process[:,:,1], time]


    for v,m,s in zip(vals, np.meshgrid(*means), std_devs):
        gbk += -(v-m.ravel()[:,None,None])**2/(2*s**2)
    return np.exp(gbk)

def gaussian_basis_Xi_matrix(process, params, num_estimators, kernel_value=None):
    dt = params['sim'][2]
    if kernel_value is None:
        kernel_value = gaussian_basis_kernel(process, num_estimators)
    return dt * np.einsum('ijk,mjk->im', kernel_value[...,:-1], kernel_value[...,:-1]) / process.shape[0]

def gaussian_basis_irr_current_average(process, params, num_estimators, kernel_value=None):
    gamma = params['gamma']
    if kernel_value is None:
        kernel_value = gaussian_basis_kernel(process, num_estimators)
    dproc = np.diff(process,axis=1)
    dw = kernel_value[...,1:] - kernel_value[...,:-1]
    current_value = kernel_value[...,:-1] * dproc[...,0] * gamma - 1/2 * dw * dproc[...,1]
    current_mean = np.sum(current_value,axis=(1,2))/process.shape[0]
    return current_mean



def get_ep(process, params, num_est, return_mats = False, kernel_params = None):

    kernel = gaussian_basis_kernel(process, num_est, kernel_params= kernel_params)

    mu = gaussian_basis_irr_current_average(process, params, num_est, kernel_value=kernel)
    Xi = gaussian_basis_Xi_matrix(process, params, num_est, kernel_value=kernel)
    
    if return_mats:
        return mu, Xi
    
    ep = mu@np.linalg.inv(Xi)@mu
    return ep
