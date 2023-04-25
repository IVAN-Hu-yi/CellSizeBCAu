import numpy as np
from scipy.integrate import solve_ivp
from parametres import Paras
from utilities import *
from initialisation import *
from odes import odes_scale_size
import matplotlib.pyplot as plt
from size_scaled_func import *
import datetime as dt
from odes import odes_not_scaled
from numpy.random import default_rng


def sim_run(N, M, para:Paras, assemblenum, tstop, teval, scale=True):

    ### initialised parametres
    # para = Paras(N, M)
    p, number = int_preferences(N, M, para.mu_c, assemblenum)
    D = int_conversion(M, para.Dbase, para.number, assemblenum, para.sparsity)
    l = int_l(M, para.l_base, assemblenum)
    rho = int_rho(M, para.rho_base, assemblenum)
    vmax = int_vmax(N, M, para.v_max_base, p, number, assemblenum)
    m = int_mt(N, para.m_base, assemblenum)
    avgm = allocate_avgm(N, para.w, assemblenum, para.ymin, para.ymax)

    ## Initialised Initial conditions
    R0 = int_R(M, para.R0, assemblenum)
    C0 = int_C(avgm, para.x0)

    # Load parametres
    para.paras(C0, R0, l, rho, p, vmax, m, D, avgm)
    time = np.linspace(0, tstop, teval)
    y0 = np.concatenate((R0, C0)).reshape(M+N,) # initial conditions

    # run and store
    if scale:
        pars = (para.l, para.m, para.rho, para.mu, para.km, para.p, para.D, para.v_in_max, para.type, para.B0, para.M0, para.E0, para.alpha, para.beta, para.gamma, para.R_half)
        result = solve_ivp(
        odes_scale_size, t_span=[time[0], time[-1]], y0=y0, t_eval=time, args=pars, dense_output=True)
    
    if not scale:
        pars = (para.l, para.m, para.rho, para.mu, para.km, para.p, para.D, para.v_in_max, para.type, para.R_half)
        result = solve_ivp(odes_not_scaled, t_span=[time[0], time[-1]], y0=y0, t_eval=time, args=pars, dense_output=True)

    Rt = result['y'][0:M]
    Ct = result['y'][M:M+N]
    t = result['t']
    # Ct = extract_Ct_single_assembly(Ct)

    return Rt, Ct, t, para

def sim_sub_run(N, M, assemblenum, tstop, teval, subcommunity_size=10, num_subcommunity=20, scale=True):
    
    ### initialised parametres
    para = Paras(N, M)
    p, number = int_preferences(N, M, para.mu_c, assemblenum)
    D = int_conversion(M, para.Dbase, para.number, assemblenum, para.sparsity)
    l = int_l(M, para.l_base, assemblenum)
    rho = int_rho(M, para.rho_base, assemblenum)
    vmax = int_vmax(N, M, para.v_max_base, p, number, assemblenum)
    m = int_mt(N, para.m_base, assemblenum)
    avgm = allocate_avgm(N, para.w, assemblenum, para.ymin, para.ymax)

    # data storage
    R_assembles = []
    C_assembles = []
    t_assembles = []
    para_assembles = []
    identifier = []

    print(f'Initialisation completed!')
    start = dt.datetime.now()

    for i in range(num_subcommunity):
        
        para = Paras(subcommunity_size, M)
        ## select corresponding species for subcommunity assemblies
        rng = default_rng(seed=assemblenum+i+8*i)
        idx = rng.choice(range(N), size=subcommunity_size, replace=False)
        identifier.append(idx)
        new_p = p[idx, :]
        new_vmax = vmax[idx, :]
        new_m = m[idx, :]
        new_avgm = avgm[idx, :]

        ## Initialised Initial conditions
        R0 = int_R(M, para.R0, assemblenum)
        C0 = int_C(new_avgm, para.x0)

       # Load parametres
        para.paras(C0, R0, l, rho, new_p, new_vmax, new_m, D, new_avgm)
        time = np.linspace(0, tstop, teval)
        y0 = np.concatenate((R0, C0)).reshape(M+subcommunity_size,) # initial conditions

        # run and store
        if scale:
            pars = (para.l, para.m, para.rho, para.mu, para.km, para.p, para.D, para.v_in_max, para.type, para.B0, para.M0, para.E0, para.alpha, para.beta, para.gamma, para.R_half, para.avgm)
            result = solve_ivp(
            odes_scale_size, t_span=[time[0], time[-1]], y0=y0, t_eval=time, args=pars, dense_output=True)
        
        if not scale:
            pars = (para.l, para.m, para.rho, para.mu, para.km, para.p, para.D, para.v_in_max, para.type, para.R_half)
            result = solve_ivp(odes_not_scaled, t_span=[time[0], time[-1]], y0=y0, t_eval=time, args=pars, dense_output=True)
        
        Rt = result['y'][0:M]
        Ct = result['y'][M:M+subcommunity_size]
        t = result['t']

        print(f'Subcommunity simulation {i+1} completed runtime:{dt.datetime.now()-start}') if (i+1)%5==0 else None

        R_assembles.append(Rt)
        C_assembles.append(Ct)
        t_assembles.append(t)
        para_assembles.append(para)

    return R_assembles, C_assembles, t_assembles, para_assembles, identifier