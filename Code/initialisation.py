import numpy as np
from parametres import Paras

# Initial Conditions
seed = 100 # set seed to ensure reproducible results in each experiment
# assemblenum: int -> set random state for each individual assembly process

def int_R(M, R, assemblenum): 
    '''
        Resource Content at t0
    '''
    # np.random.seed(seed+assemblenum)
    return np.full((M, 1), R)

def allocate_avgm(N, w, assemblenum, ymin, ymax):
    '''Initialised initial cell mass

    Args:
        N (_type_): _description_
        w (_type_): _description_
        assemblenum (_type_): _description_
        ymin (_type_): _description_
        ymax (_type_): _description_
    
    Returns:
        np.array: average mass array of a community
    '''
    np.random.seed(seed+assemblenum)
    y = np.random.beta(1, w, N).reshape(N, 1)
    return 10 ** (ymin + (ymax-ymin)*y)

def int_C(avgm, density):
    '''
        Biomass Content drawn from beta distribution at t0
    '''
    
    return density * avgm

# define parametres

def int_preferences(N, M, mu_c, assemblenum):
    
    '''Guassian sampling of preferences, assume all are generalists

    Returns:
        np.array: N*M matries
        number: number of preferred resources
    '''

    p = np.zeros((N, M))
    number = int(mu_c*M) if int(mu_c*M) > 0 else 1 # number of preferred resources

    for i in range(N):
        np.random.seed(i) # ensure for each experiment, each species same preferences
        idx = np.random.randint(0, M, number) # select favored resoruces
        
        np.random.seed(i)
        values = np.random.normal(1/number, 0.001, number).tolist() # initialisation
        for x, j in zip(idx, range(len(values))):
            p[i, x] = values[j]  # assign values
        p[i, :] = p[i, :]/np.sum(p[i, :]) # normalised to 1

    return (p, number)

def int_conversion(M, Dbase, number, assemblenum, sparse=False):

    '''guassian sampling of conversion around Dbase, only allow sysmetrical matrix

    Args:
        M (int): int
        Dbase (float): Guassian mean

    Returns:
        np.array: shape input_resource * output resource 
    '''

    # np.random.seed(seed+assemblenum)
    np.random.seed(seed+100)
    if sparse: 
        D = np.random.normal(Dbase, Dbase/10, (M, M)).reshape(M, M) # sample conversion
        D = D * (1-np.tri(*D.shape, k=-1)) # not allow reversible reactions
    else:
        D = np.zeros((M, M))
        for i in range(M):
            np.random.seed(seed+i*100)
            num = np.random.randint(1, number+1) # number of resource type being converted from one resource
            idx = np.random.randint(0, M, num) # select idx for corresponding resource type
            values = np.random.normal(Dbase, Dbase/10, num) # sample conversion
            # D = D * (1-np.tri(*D.shape, k=-1)) # not allow reversible reactions
            for x, y in zip(idx, range(num)):
                D[i, x] = values[y]

    return D/np.sum(D, axis=1)[:, np.newaxis] # row-wise normalisation
    
def int_l(M, l, assemblenum, same=True):
    '''
     return leakage
    '''
    np.random.seed(seed+assemblenum)
    if same: return np.array([l]*M).reshape(M, 1)
    else: return np.random.normal(l, l/10, M).reshape(M, 1)

def int_rho(M, rho, assemblenum):
    '''
    define external resource supply
    '''
    # np.random.seed(seed)
    return np.array([rho]*M).reshape(M, 1)

def int_vmax(N, M, v_max_base, p, number, assemblenum):

    '''initialised maximum uptake of each resource 

    Args:
        v_max_base: mean vmax for uptake
        p: preferences for identify preferred resource
        number: number of preferred resources

    Returns:
        np.array : N by M matrix
    '''

    vmax = np.ones((N, M))*0.1 # for non-favored 0.1 max uptake
    for i in range(N):
        temp_p = p[i, :] # identify preferred resource
        temp_vmax = vmax[i, :]
        temp_p_copy = np.sort(temp_p)[::-1]
        val = temp_p_copy[number-1] # least preferred resource among the preferred
        np.random.seed(i+40+assemblenum)
        temp_vmax[temp_p>=val] = np.random.normal(v_max_base, 0.1, len(temp_vmax[temp_p>=val])) # define max uptake
        vmax[i, :] = temp_vmax # update temp_vmax
    
    return vmax

def int_mt(N, m, assemblenum):
    '''initialise maitainence

    Returns:
        np.array : N*1 matrix
    '''
    # np.random.seed(seed+assemblenum)
    np.random.seed(seed)
    return np.random.normal(m, 0.1, (N, 1)).reshape(N, 1)
