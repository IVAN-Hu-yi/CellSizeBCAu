class Paras:
    '''
        define model parametres
    '''

    def __init__(self, N, M):

        # basic
        self.N = N # number of species
        self.M = M # number of resource types

        ####### Model Paras fixed

        self.R_half = 1 # coefficient in sigma func
        self.mu = 1 # intrinsic growth rate  mass^-1
        self.km = 1 # individual per unit mass mass^-1
        self.rho_base = 0.1 # mean for rho
        self.l_base = 0.1 # mean for leakage
        self.m_base = 1.2 # mean for m
        self.v_max_base = 1.5 # mean for vmax
        self.type = 2
        self.x0 = 10 # initial density mass/individual

        ####### Scaling Paras

        self.B0 = 1e-03 # Normalisation constant for resource inflow
        self.M0 = 1e-03 # Normalisation constant for maintenance
        self.E0 = 1e-03 # Normalisation constant for outflow
        self.alpha = - 0.25 # size-scaling exponent for inflow 
        self.gamma = 0.86 # size-scaling exponent for outflow
        self.beta = 0.75 # maintainence
        self.ymin = 0.7
        self.ymax = 2


        ####### initialisation
        self.R0 = 10 # initial resource mass
        self.w = 2 # one parameter distribution 1 or 2 B(1, w)

        ####### relevant paras in consumer preferences (similar to Marsland 2019)
        self.mu_c = 0.25 # proportion of favor resources types
        self.c0 = 0.01 # fraction of uptake of non-favored resource type

        ####### relevant paras in conversion efficiency
        self.Dbase = 0.2 # base efficiency for all resource-resource pair 
        self.number = 15 # integer indicating maximum number of resources being converted
        self.sparsity = False # whether limited converted resource applied

    def paras(self, Ci, Ri, l, rho, p, vmax, m, D, avgm):

        ####### Model Paras defined
        self.C = Ci # initial biomass for each species N*1 array
        self.R = Ri # initial resource content for each resource type M*1 array
        self.l = l # leakage fraction for each resource M*1 matrix
        self.rho = rho # external resource supply M*1 array
        self.p = p # resource preferences N*M matrix
        self.v_in_max = vmax # max uptake/sigma func N*M matrix
        self.m = m # maintainence N*1 array
        self.D = D # conversion effciency
        self.avgm = avgm # average individual cell size N*1 matrix