## define model functions

from initialisation import *
from size_scaled_func import *
import numpy as np
from utilities import *

def vin(p, R, Rhalf, vmax=None, type=2):
    '''Calcualte Vin for each species

    Args:
        p (N by M matrix): preference 
        R (M by 1 matrix): resource concentration
        Rhalf ( float ): half-velocity constant
        vmax (N by M matrix, optional): maximum uptake rate. Defaults to None.
        type (int, optional): Type I or Type II functional response. Defaults to 2.

    Returns:
       N by 1 vector: Vin for each species
    '''
    # calculate uptake
    uptake = p * R.T[:, np.newaxis]
    uptake = uptake.reshape(p.shape)

    if type==2: # Monod function
        if vmax is None:
            print('Vmax required for monod function')
        else:
            return vmax*((uptake)/(Rhalf + uptake))

    elif type==1: # linear
        return uptake
    
def vgrow(vin, leakage):
    '''calculate resource flow needed for growth

    Args:
        vin (np.array): N*M matrix for resource inflow
        leakage (np.array): M*1 matrix -- leakage fraction

    Returns:
        N*1 vector: resource flow needed for growth

    '''
    return vin @ (1-leakage)

def vout(vin, leakage, D):
     
    '''caculate vout
    
    Args:
        vin (np.array): N*M matrix for resource inflow
        leakage (np.array): M*1 matrix -- leakage fraction
        D (np.array): N*M matrix for conversion efficiency

    Returns:
        np.array: N*M matrix
    '''

    vout = (vin @ D)* leakage.T[:, np.newaxis]
    vout = vout.reshape(vin.shape)
    return vout

def odes_not_scaled(t, y, l, m, rho, mu, km, p, D, vmax, type, Rhalf):
    '''ODEs of our model -- non-scaled version

    Args:
        y (list): integrate over
        t (scaler/vector): dummies required for ode
        l (np.array): leakge M*1 matrix
        m (np.array): maintainence N*1 matrix 
        rho (np.array): external resource supply M*1
        mu (float): growth rate
        km (float): normalmisation constant
        p (np.array): preferences N*M matrix
        D (np.array): conversion efficiency M*M
        vmax (np.array): maximum uptake N*M matrix
        type (int): type of functional response 

    Returns:
        list: [dC/dt, dR/dt]
    '''
    N, M = vmax.shape
    R, C = y[0:M], y[M:M+N]

    R[R<0 | R==0] = 0 
    C[C<0 | C==0] = 0

    v_in = vin(p, R, Rhalf, vmax, type)
    v_grow = vgrow(v_in, l)
    v_out = vout(v_in, l, D)
    vdiff = v_out - v_in



    A = np.empty((N+M))
    drdt = rho+km*(vdiff.T @ C)
    A[0:M] = drdt.reshape(M,)
    dcdt = mu*C*(v_grow-m)
    A[M:M+N] = dcdt.reshape(N,)

    return [mu*C*(v_grow-m), rho+km*(vdiff.T @ C)]

def odes_scale_size(t, y, l, m, rho, mu, km, p, D, vmax, type, B0, M0, E0, alpha, beta, gamma, Rhalf, avgm):
    '''ODEs of our model -- scaled version

    Args:
        y (list): initial values
        t (scaler/vector): dummies required for ode
        l (np.array): leakge M*1 matrix
        m (np.array): maintainence N*1 matrix 
        rho (np.array): external resource supply M*1
        mu (float): growth rate
        km (float): normalmisation constant
        p (np.array): preferences N*M matrix
        D (np.array): conversion efficiency M*M
        vmax (np.array): maximum uptake N*M matrix
        type (int): type of functional response 
        B0 (float): normalmisation constant
        M0 (float): normalmisation constant
        E0 (float): normalmisation constant
        alpha (float): scaling constant
        beta (float) : scaling constant
        gamma (float): scaling constant
        Rhalf (float): constant for sigma function
        avgm (np.array): N*1 matrix containing the average cell size of a population
    Returns:
        A (np.array): A[0:M] = dR/dt; A[N:M+N] = dC/dt
    '''
    N, M = vmax.shape
    R, C = y[0:M], y[M:M+N]

    R[R<0] = 0 
    C[C<0] = 0
    avgm = avgm.reshape(N,)
    # C[C - 0.3 * avgm < 0] = 0
    
    R = R.reshape(M, 1)
    C = C.reshape(N, 1)

    # caculate intermediate
    avgm = avgm.reshape(N, 1)
    vmax = scale_vmax(vmax, avgm, B0, alpha)
    v_in = vin(p, R, Rhalf, vmax, type)
    v_grow = vgrow(v_in, l)
    v_out = vout(v_in, l, D)
    # v_out = scale_vout(v_out, avgm, E0, gamma)
    vdiff = v_out - v_in
    m_scale = scale_mt(m, avgm, M0, beta)

    # construct equations
    A = np.empty((N+M))
    drdt = rho+km*(vdiff.T @ C)
    A[0:M] = drdt.reshape(M,)
    dcdt = mu*C*(v_grow-m_scale)
    A[M:M+N] = dcdt.reshape(N,)

    return A