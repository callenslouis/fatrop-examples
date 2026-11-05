from casadi import *
import numpy as np

from reformulator import Reformulator

class OptiGenerator():
    def __init__(self, introduce_periodicity_vars=False):
        self.n_coll = 3
        self.casadi_func_folder = f'casadi_funcs/n_coll_{self.n_coll}'
        self.introduce_periodicity_vars = introduce_periodicity_vars
        self.load_casadi_funcs()
        
    # scale_constraints: every equality constraint row, except the explicit dynamics rows (identity in x_{k+1}),
    # is divided by the inf-norm of its Jacobian row at the initial guess. The same rule is applied in generate() and
    # generate_implicit(), so physically identical rows get identical scale factors (inf-norm equilibration removes
    # constant factors such as the qdot scaling of the reformulator). Row scaling does not change the solution, but it
    # changes the pivots of the stage-wise LU factorizations in fatrop, which use an absolute rank tolerance
    # (linsol_lu_fact_tol) and are borderline for this problem near the end of the horizon.
    def generate(self, scale_constraints=False):
        self.rewrite_problem_formulation()
        return self.generate_opti(scale_constraints)

    @staticmethod
    def row_scale(jac_fun, args):
        J = np.abs(np.array(jac_fun(*args)))
        row_max = J.max(axis=1) if J.shape[1] > 0 else np.zeros(J.shape[0])
        return DM(1.0/np.where(row_max > 0, row_max, 1.0))
        
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
        
        # visualize
        # self.reformulator.show_gk_jacobian_structure_for_debugging(self.f_gk_reformulated)
        # print(self.get_nb_dynamics_constraints())
        # print(self.get_nb_zk_vars())
        # print(self.uk_orig.shape[0])
        # exit()
        
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
        
    def generate_opti(self, scale_constraints=False):
        uk = SX.sym('uk', self.uu_init_reordered.shape[0])
        s_path = [None]*self.N
        s_K = None
        if scale_constraints:
            J_gk = Function('J_gk', [self.xk, uk], [densify(jacobian(self.f_gk_reformulated(self.xk, uk), vertcat(self.xk, uk)))])
            J_g0 = Function('J_g0', [self.xk, uk], [densify(jacobian(self.f_g0_reformulated(self.xk, uk), vertcat(self.xk, uk)))])
            J_gK = Function('J_gK', [self.xk], [densify(jacobian(self.f_gK(self.xk), self.xk))])
            for k in range(self.N):
                s_path[k] = self.row_scale(J_g0 if k == 0 else J_gk, (self.xx_init[:, k], self.uu_init_reordered[:, k]))
            s_K = self.row_scale(J_gK, (self.xx_init[:, self.N],))

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
            g_path = self.f_g0_reformulated(xx[k], uu[k]) if k == 0 else self.f_gk_reformulated(xx[k], uu[k])
            opti.subject_to((g_path if s_path[k] is None else s_path[k]*g_path) == 0)

            # inequality constraint
            opti.subject_to(self.lb <= (self.f_gk_ineq(xx[k], uu[k]) <= self.ub))

            # initial guess
            opti.set_initial(xx[k], self.xx_init[:, k])
            opti.set_initial(uu[k], self.uu_init_reordered[:, k])

            # final constraints
            if k == self.N-1:
                g_K = self.f_gK(xx[k+1])
                opti.subject_to((g_K if s_K is None else s_K*g_K) == 0)
                opti.set_initial(xx[k+1], self.xx_init[:, k+1])
                
            # objective
            obj += self.f_objk(xx[k], uu[k])
            
        opti.minimize(obj)
        # opti.minimize(sumsqr(hcat(uu)))
        
        opts_casadi = {'expand': True, 'detect_simple_bounds': True, 
               'structure_detection': 'auto'}
        opts_fatrop = {'tol': 1e-5, 'mu_init': 0.1, 'max_iter': 300, 'problem_type': 'ocp_type'}
        
        return opti, xx, uu, opts_casadi, opts_fatrop
    
    # Formulation for the implicit Riccati recursion (Fatrop problem_type 'implicit_ocp_type'). With Radau collocation,
    # the last collocation point is the end of the interval, so q_coll(n_coll) = q_{k+1} and qdot_coll(n_coll) =
    # qdot_{k+1}: these are not introduced as controls, the stage functions use the next states instead.
    #   states:   x_k = [q_k, qdot_k, psi_k]
    #   controls: u_k = [act_mesh, q_coll(1), qdot_coll(1), qddot_coll(1), ..., q_coll(n_coll-1), qdot_coll(n_coll-1),
    #                    qddot_coll(n_coll-1), qddot_coll(n_coll)]
    #   dynamics: F_k(x_k, u_k, x_{k+1}) = [collocation equations of q and qdot; system dynamics at the last
    #             collocation point; psi_k - psi_{k+1}] = 0, the implicit states (q, qdot) and equations come first
    #   path:     system dynamics at the other collocation points (and the periodicity constraints at stage 0)
    def generate_implicit(self, scale_constraints=False):
        assert not self.introduce_periodicity_vars, "The implicit formulation propagates psi as a state"
        n_q, n_coll = self.n_coords, self.n_coll
        n_qx = 2*n_q  # states that appear implicitly: q and qdot
        xk = self.xk
        xkp = SX.sym('xkp', self.nx)

        # controls of the implicit formulation
        act = SX.sym('act_mesh_k', self.n_act_mesh)
        q_c = [SX.sym(f'q_coll_{i+1}', n_q) for i in range(n_coll-1)]
        qdot_c = [SX.sym(f'qdot_coll_{i+1}', n_q) for i in range(n_coll-1)]
        qddot_c = [SX.sym(f'qddot_coll_{i+1}', n_q) for i in range(n_coll)]
        uk = act
        for i in range(n_coll-1):
            uk = vertcat(uk, q_c[i], qdot_c[i], qddot_c[i])
        uk = vertcat(uk, qddot_c[n_coll-1])
        nu = uk.shape[0]

        # original controls, with the last collocation point replaced by the next states
        uk_orig = vertcat(act,
                          vertcat(*q_c, xkp[:n_q]),
                          vertcat(*qdot_c, xkp[n_q:n_qx]),
                          vertcat(*qddot_c))
        assert uk_orig.shape[0] == self.nu

        # rows of f_gk: collocation equations of q and qdot, then the system dynamics per collocation point
        gk = self.f_gk_orig(xk, uk_orig)
        n_rows_coll = 2*n_coll*n_q
        sysdyn = [gk[n_rows_coll + i*n_q:n_rows_coll + (i+1)*n_q] for i in range(n_coll)]
        f_dyn = Function('f_dyn', [xk, uk, xkp],
                         [vertcat(gk[:n_rows_coll], sysdyn[n_coll-1], xk[n_qx:] - xkp[n_qx:])])
        # the path constraints only depend on x_k and u_k (Function raises an error otherwise)
        f_gk_path = Function('f_gk_path', [xk, uk], [vertcat(*sysdyn[:n_coll-1])])
        f_g0_path = Function('f_g0_path', [xk, uk], [vertcat(self.f_g0_orig_unique(xk, uk_orig), *sysdyn[:n_coll-1])])
        f_objk = Function('f_objk', [xk, uk, xkp], [self.f_objk_orig(xk, uk_orig)])

        # all inequalities are bounds on single variables; the bounds on the last collocation point are the bounds
        # on the next states, so they are left out at stage k and the bounds on x_N are added at stage N
        g_ineq = self.f_gk_ineq_orig(xk, uk_orig)
        rows_xkp = set(jacobian(g_ineq, xkp).sparsity().row())
        rows_u = set(jacobian(g_ineq, uk).sparsity().row())
        rows_k = [r for r in range(g_ineq.shape[0]) if r not in rows_xkp]  # bounds on x_k and u_k
        rows_x = [r for r in rows_k if r not in rows_u]                     # bounds on x_k
        f_gk_ineq = Function('f_gk_ineq', [xk, uk], [g_ineq[rows_k]])
        f_gK_ineq = Function('f_gK_ineq', [xk], [g_ineq[rows_x]])
        lb_k, ub_k = self.lb[rows_k], self.ub[rows_k]
        lb_K, ub_K = self.lb[rows_x], self.ub[rows_x]

        # initial guess of the controls
        u_init = [self.uu_init[:self.n_act_mesh, :]]
        q_off, qdot_off, qddot_off = self.n_act_mesh, self.n_act_mesh + n_coll*n_q, self.n_act_mesh + 2*n_coll*n_q
        for i in range(n_coll-1):
            u_init += [self.uu_init[q_off + i*n_q:q_off + (i+1)*n_q, :],
                       self.uu_init[qdot_off + i*n_q:qdot_off + (i+1)*n_q, :],
                       self.uu_init[qddot_off + i*n_q:qddot_off + (i+1)*n_q, :]]
        u_init += [self.uu_init[qddot_off + (n_coll-1)*n_q:qddot_off + n_coll*n_q, :]]
        uu_init = vertcat(*u_init)
        assert uu_init.shape[0] == nu

        # constraint scaling (see generate()): the implicit dynamics rows and the path/terminal equality rows are
        # scaled, the explicit dynamics rows (psi) keep their -I structure. The Jacobians are evaluated with the q and
        # qdot of x_{k+1} set to the initial guess of the last collocation point, which is the point at which
        # generate() evaluates the same rows.
        n_impl = n_rows_coll + n_q
        ones_x = DM.ones(self.nx - n_qx)
        s_dyn = [DM.ones(n_impl + self.nx - n_qx)]*self.N
        s_path = [None]*self.N
        s_K = None
        if scale_constraints:
            row_scale = self.row_scale
            fdyn_impl = f_dyn(xk, uk, xkp)[:n_impl]
            J_dyn = Function('J_dyn', [xk, uk, xkp], [densify(jacobian(fdyn_impl, vertcat(xk, uk, xkp)))])
            J_gk = Function('J_gk', [xk, uk], [densify(jacobian(f_gk_path(xk, uk), vertcat(xk, uk)))])
            J_g0 = Function('J_g0', [xk, uk], [densify(jacobian(f_g0_path(xk, uk), vertcat(xk, uk)))])
            xK = SX.sym('xK', self.nx)
            J_gK = Function('J_gK', [xK], [densify(jacobian(self.f_gK_orig(xK), xK))])
            for k in range(self.N):
                xk0, uk0 = self.xx_init[:, k], uu_init[:, k]
                q_last = self.uu_init[q_off + (n_coll-1)*n_q:q_off + n_coll*n_q, k]
                qdot_last = self.uu_init[qdot_off + (n_coll-1)*n_q:qdot_off + n_coll*n_q, k]
                xkp0 = vertcat(q_last, qdot_last, self.xx_init[n_qx:, k+1])
                s_dyn[k] = vertcat(row_scale(J_dyn, (xk0, uk0, xkp0)), ones_x)
                s_path[k] = row_scale(J_g0 if k == 0 else J_gk, (xk0, uk0))
            s_K = row_scale(J_gK, (self.xx_init[:, self.N],))

        # the variables are ordered per stage as (x_k, u_k), the constraints of a stage start with its dynamics
        opti = Opti()
        xx = []
        uu = []
        for k in range(self.N):
            xx.append(opti.variable(self.nx, 1))
            uu.append(opti.variable(nu, 1))
        xx.append(opti.variable(self.nx, 1))

        obj = 0
        for k in range(self.N):
            opti.subject_to(s_dyn[k]*f_dyn(xx[k], uu[k], xx[k+1]) == 0)
            g_path = f_g0_path(xx[k], uu[k]) if k == 0 else f_gk_path(xx[k], uu[k])
            opti.subject_to((g_path if s_path[k] is None else s_path[k]*g_path) == 0)
            opti.subject_to(lb_k <= (f_gk_ineq(xx[k], uu[k]) <= ub_k))

            opti.set_initial(xx[k], self.xx_init[:, k])
            opti.set_initial(uu[k], uu_init[:, k])

            obj += f_objk(xx[k], uu[k], xx[k+1])

        g_K = self.f_gK_orig(xx[self.N])
        opti.subject_to((g_K if s_K is None else s_K*g_K) == 0)
        opti.subject_to(lb_K <= (f_gK_ineq(xx[self.N]) <= ub_K))
        opti.set_initial(xx[self.N], self.xx_init[:, self.N])
        opti.minimize(obj)

        # the stage sizes tell the structure detection which variables are states
        opts_casadi = {'expand': True, 'detect_simple_bounds': True, 'structure_detection': 'manual',
                       'nx': [self.nx]*(self.N+1), 'nu': [nu]*self.N + [0]}
        opts_fatrop = {'tol': 1e-5, 'mu_init': 0.1, 'max_iter': 300, 'problem_type': 'implicit_ocp_type',
                       'linsol_nb_of_implicit_states': n_qx}

        return opti, xx, uu, opts_casadi, opts_fatrop

    def get_nb_dynamics_constraints(self):
        return self.reformulator.number_of_constraints_with_zk
    
    def get_nb_zk_vars(self):
        return self.zk.shape[0]