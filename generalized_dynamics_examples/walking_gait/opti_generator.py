from casadi import *

from reformulator import Reformulator

class OptiGenerator():
    def __init__(self, introduce_periodicity_vars=False):
        self.n_coll = 3
        self.casadi_func_folder = f'casadi_funcs/n_coll_{self.n_coll}'
        self.introduce_periodicity_vars = introduce_periodicity_vars
        self.load_casadi_funcs()
        
    def generate(self):
        self.rewrite_problem_formulation()
        return self.generate_opti()
        
    def rewrite_problem_formulation(self):
        self.reorder_uk_vars()
        self.update_path_constraints()
        self.update_initial_guess()

    def load_casadi_funcs(self):
        self.f_objk_orig = Function.load(f"{self.casadi_func_folder}/f_objk.casadi")
        self.f_g0_orig = Function.load(f"{self.casadi_func_folder}/f_g0.casadi")
        self.f_gk_orig = Function.load(f"{self.casadi_func_folder}/f_gk.casadi")
        self.f_gK_orig = Function.load(f"{self.casadi_func_folder}/f_gN.casadi")
        self.f_gk_ineq_orig = Function.load(f"{self.casadi_func_folder}/f_gk_ineq.casadi")
        self.f_gap_orig = Function.load(f"{self.casadi_func_folder}/f_gap.casadi")
        self.f_lb_orig = Function.load(f"{self.casadi_func_folder}/f_lb.casadi")
        self.f_ub_orig = Function.load(f"{self.casadi_func_folder}/f_ub.casadi")
        self.f_x_init_orig = Function.load(f"{self.casadi_func_folder}/f_init_x.casadi")
        self.f_u_init_orig = Function.load(f"{self.casadi_func_folder}/f_init_u.casadi")
        self.f_g0_orig_unique = Function.load(f"{self.casadi_func_folder}/f_g0_unique.casadi")

        # derive dimensions
        self.xx_init = self.f_x_init_orig()['x_init']
        self.uu_init = self.f_u_init_orig()['u_init']
        self.lb = self.f_lb_orig()['lb']
        self.ub = self.f_ub_orig()['ub']
        self.nx = self.xx_init.shape[0]
        self.nu = self.uu_init.shape[0]
        self.N = self.xx_init.shape[1] - 1
        self.xk = SX.sym('xk', self.nx)
        
        self.n_act_mesh = 18
        self.n_coords = 9
        self.dt = (1.0/0.9/2.0)/self.N
        
    # Function to reorder uk variables to fit the desired structure
    def reorder_uk_vars(self):
        # from
        # [act_mesh_k_SX, q_coll(1)_k_SX, ..., q_coll(3)_k_SX, qdot_coll(1)_k_SX, ..., qdot_coll(3)_k_SX, qddot_coll(1)_k_SX, ..., qddot_coll(3)_k_SX]
        #                                      --------------                          ------------------                           
        # to
        # [act_mesh_k_SX, q_coll(1)_k_SX, ..., qddot_coll(1)_k_SX, q_coll(2)_k_SX, ..., qddot_coll(2)_k_SX, qddot_coll(3)_k_SX, q_coll(3)_k_SX, ...]
        #                                                                                                                       --------------------
        # such that zk variables are at the back
        self.act_mesh = SX.sym('act_mesh_k', self.n_act_mesh)
        self.q_coll = SX.sym('q_coll_k', self.n_coll * self.n_coords)
        self.qdot_coll = SX.sym('qdot_coll_k', self.n_coll * self.n_coords)
        self.qddot_coll = SX.sym('qddot_coll_k', self.n_coll * self.n_coords)
        self.uk_orig = vertcat(self.act_mesh, self.q_coll, self.qdot_coll, self.qddot_coll)
        assert self.uk_orig.shape[0] == self.nu

        periodicity_vars = SX.sym('periodicity_vars', self.introduce_periodicity_vars*(self.n_coords * 2 - 1)) # all q and qdot variables except for the forward foot x position
        self.zk = vertcat(self.q_coll[(self.n_coll-1)*self.n_coords:self.n_coll*self.n_coords], self.qdot_coll[(self.n_coll-1)*self.n_coords:self.n_coll*self.n_coords], periodicity_vars)
        if self.introduce_periodicity_vars:
            assert self.zk.shape[0] == self.nx
            periodicity_constraint = periodicity_vars - self.xk[self.n_coords * 2:]
        else:
            periodicity_constraint = SX.zeros(0,1)

        uk = self.act_mesh
        for i in range(self.n_coll-1):
            uk = vertcat(uk, self.q_coll[i*self.n_coords:(i+1)*self.n_coords], self.qdot_coll[i*self.n_coords:(i+1)*self.n_coords], self.qddot_coll[i*self.n_coords:(i+1)*self.n_coords])
        uk = vertcat(uk, self.qddot_coll[(self.n_coll-1)*self.n_coords:self.n_coll*self.n_coords], self.zk) # keep the last q_coll and qdot_coll as zk variables to enforce periodicity
        self.uk = uk
        
        self.f_objk = Function('f_objk', [self.xk, self.uk], [self.f_objk_orig(self.xk, self.uk_orig)])
        self.f_g0 = Function('f_g0', [self.xk, self.uk], [vertcat(self.f_g0_orig(self.xk, self.uk_orig), periodicity_constraint)])
        self.f_gk = Function('f_gk', [self.xk, self.uk], [vertcat(self.f_gk_orig(self.xk, self.uk_orig), periodicity_constraint)])
        self.f_gK = Function('f_gK', [self.xk], [self.f_gK_orig(self.xk)])
        self.f_gk_ineq = Function('f_gk_ineq', [self.xk, self.uk], [self.f_gk_ineq_orig(self.xk, self.uk_orig)])
        self.f_gap = Function('f_gap', [self.xk, self.uk], [self.zk if self.introduce_periodicity_vars else vertcat(self.zk, self.xk[self.n_coords*2:])])
        self.f_g0_unique = Function('f_g0_unique', [self.xk, self.uk], [self.f_g0_orig_unique(self.xk, self.uk_orig)])
        
    # Function to reformulate path constraints and dynamics to fit the structure-exploiting structure
    def update_path_constraints(self):
        self.reformulator = Reformulator(self.n_coll, self.n_coords, 
                                    self.n_act_mesh, self.dt, self.xk, self.uk,
                                    self.uk_orig, self.q_coll, self.qdot_coll, 
                                    self.qddot_coll)
        
        self.f_gk_reformulated = self.reformulator.get_f_gk_reformulated(self.f_gk)
        self.f_g0_reformulated = Function('f_g0_reformulated', [self.xk, self.uk], 
                                     [vertcat(
                                         self.f_g0_unique(self.xk, self.uk), 
                                         self.f_gk_reformulated(self.xk, self.uk))])
        
    # updte initial guess of controls to fit the reordered structure
    def update_initial_guess(self):
        act_init = self.uu_init[:self.n_act_mesh, :]
        q_coll_init = self.uu_init[self.n_act_mesh:self.n_act_mesh+self.n_coll*self.n_coords, :]
        qdot_coll_init = self.uu_init[self.n_act_mesh+self.n_coll*self.n_coords:self.n_act_mesh+2*self.n_coll*self.n_coords, :]
        qddot_coll_init = self.uu_init[self.n_act_mesh+2*self.n_coll*self.n_coords:self.n_act_mesh+3*self.n_coll*self.n_coords, :]

        # actual controls
        uu_init_reordered = act_init

        # collocation points
        for i in range(self.n_coll-1):
            uu_init_reordered = vertcat(uu_init_reordered, q_coll_init[i*self.n_coords:(i+1)*self.n_coords, :], qdot_coll_init[i*self.n_coords:(i+1)*self.n_coords, :], qddot_coll_init[i*self.n_coords:(i+1)*self.n_coords, :])

        # last collocation point qddot and zk values
        self.uu_init_reordered = vertcat(uu_init_reordered, 
                                    qddot_coll_init[(self.n_coll-1)*self.n_coords:self.n_coll*self.n_coords, :], 
                                    q_coll_init[(self.n_coll-1)*self.n_coords:self.n_coll*self.n_coords, :], 
                                    qdot_coll_init[(self.n_coll-1)*self.n_coords:self.n_coll*self.n_coords, :])
        if self.introduce_periodicity_vars:
            self.uu_init_reordered = vertcat(self.uu_init_reordered, self.xx_init[self.n_coords * 2:, :-1]) # periodicity vars initial guess

        assert self.uu_init_reordered.shape[0] == self.nu + (2*self.n_coords - 1 if self.introduce_periodicity_vars else 0)
        
    def generate_opti(self):
        opti = Opti()
        xx = []
        uu = []
        for k in range(self.N):
            xx.append(opti.variable(self.nx, 1))
            uu.append(opti.variable(self.nu + (2*self.n_coords - 1 if self.introduce_periodicity_vars else 0), 1))
        xx.append(opti.variable(self.nx, 1))  # add state at mesh point N

        obj = 0
        for k in range(self.N):
            # gap-closing constraint
            opti.subject_to(xx[k+1] == self.f_gap(xx[k], uu[k]))
            
            # # equality constraint
            if k == 0:
                opti.subject_to(self.f_g0_reformulated(xx[k], uu[k]) == 0)
            else:
                opti.subject_to(self.f_gk_reformulated(xx[k], uu[k]) == 0)

            # inequality constraint
            opti.subject_to(self.lb <= (self.f_gk_ineq(xx[k], uu[k]) <= self.ub))

            # initial guess
            opti.set_initial(xx[k], self.xx_init[:, k])
            opti.set_initial(uu[k], self.uu_init_reordered[:, k])

            # final constraints
            if k == self.N-1:
                opti.subject_to(self.f_gK(xx[k+1]) == 0)
                opti.set_initial(xx[k+1], self.xx_init[:, k+1])
                
            # objective
            obj += self.f_objk(xx[k], uu[k])
            
        opti.minimize(obj)
        # opti.minimize(sumsqr(hcat(uu)))
        
        opts_casadi = {'expand': True, 'detect_simple_bounds': True, 
               'structure_detection': 'auto'}
        opts_fatrop = {'tol': 1e-5, 'mu_init': 0.1, 'max_iter': 300, 'problem_type': 'ocp_type'}
        
        return opti, xx, uu, opts_casadi, opts_fatrop
    
    def get_nb_dynamics_constraints(self):
        return self.reformulator.number_of_constraints_with_zk
    
    def get_nb_zk_vars(self):
        return self.zk.shape[0]