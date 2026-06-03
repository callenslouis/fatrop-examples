from casadi import *
from collocation_scheme import collocation_scheme
from visualizer import visualize_gait_cycle
from reformulator import Reformulator
import matplotlib.pyplot as plt
import numpy as np
import pickle as pkl
import yaml
import sys
import io

######################
### Set parameters ###
######################
config_file = "config.yaml"
with open(config_file, 'r') as f:
    config = yaml.safe_load(f)

print(config)

n_coll = config['n_coll']
introduce_periodicity_vars = config['introduce_periodicity_vars']
problem_type = config['problem_type']
save_opti_f = config['save_opti_f']
solve = True
store_solution = config['store_solution']
load_solution = config['load_solution']
visualize_solution = config['visualize_solution']

######################
### load functions ###
######################
folder = f"casadi_funcs/n_coll_{n_coll}"
f_objk_orig = Function.load(f"{folder}/f_objk.casadi")
f_g0_orig = Function.load(f"{folder}/f_g0.casadi")
f_gk_orig = Function.load(f"{folder}/f_gk.casadi")
f_gK_orig = Function.load(f"{folder}/f_gN.casadi")
f_gk_ineq_orig = Function.load(f"{folder}/f_gk_ineq.casadi")
f_gap_orig = Function.load(f"{folder}/f_gap.casadi")
f_lb_orig = Function.load(f"{folder}/f_lb.casadi")
f_ub_orig = Function.load(f"{folder}/f_ub.casadi")
f_x_init_orig = Function.load(f"{folder}/f_init_x.casadi")
f_u_init_orig = Function.load(f"{folder}/f_init_u.casadi")
f_g0_orig_unique = Function.load(f"{folder}/f_g0_unique.casadi")

# derive dimensions
xx_init = f_x_init_orig()['x_init']
uu_init = f_u_init_orig()['u_init']
lb = f_lb_orig()['lb']
ub = f_ub_orig()['ub']
nx = xx_init.shape[0]
nu = uu_init.shape[0]
N = xx_init.shape[1] - 1
xk = SX.sym('xk', nx)

#############################
### change ordering of uk ###
#############################
# from
# [act_mesh_k_SX, q_coll(1)_k_SX, ..., q_coll(3)_k_SX, qdot_coll(1)_k_SX, ..., qdot_coll(3)_k_SX, qddot_coll(1)_k_SX, ..., qddot_coll(3)_k_SX]
#                                      --------------                          ------------------                           
# to
# [act_mesh_k_SX, q_coll(1)_k_SX, ..., qddot_coll(1)_k_SX, q_coll(2)_k_SX, ..., qddot_coll(2)_k_SX, qddot_coll(3)_k_SX, q_coll(3)_k_SX, ...]
#                                                                                                                       --------------------
# such that zk variables are at the back
n_act_mesh = 18
n_coords = 9
dt = (1.0/0.9/2.0)/N
act_mesh = SX.sym('act_mesh_k', n_act_mesh)
q_coll = SX.sym('q_coll_k', n_coll * n_coords)
qdot_coll = SX.sym('qdot_coll_k', n_coll * n_coords)
qddot_coll = SX.sym('qddot_coll_k', n_coll * n_coords)
uk_orig = vertcat(act_mesh, q_coll, qdot_coll, qddot_coll)
assert uk_orig.shape[0] == nu

periodicity_vars = SX.sym('periodicity_vars', introduce_periodicity_vars*(n_coords * 2 - 1)) # all q and qdot variables except for the forward foot x position
zk = vertcat(q_coll[(n_coll-1)*n_coords:n_coll*n_coords], qdot_coll[(n_coll-1)*n_coords:n_coll*n_coords], periodicity_vars)
if introduce_periodicity_vars:
    assert zk.shape[0] == nx
    periodicity_constraint = periodicity_vars - xk[n_coords * 2:]
else:
    periodicity_constraint = SX.zeros(0,1)

uk = act_mesh
for i in range(n_coll-1):
    uk = vertcat(uk, q_coll[i*n_coords:(i+1)*n_coords], qdot_coll[i*n_coords:(i+1)*n_coords], qddot_coll[i*n_coords:(i+1)*n_coords])
uk = vertcat(uk, qddot_coll[(n_coll-1)*n_coords:n_coll*n_coords], zk) # keep the last q_coll and qdot_coll as zk variables to enforce periodicity
# uk  = vertcat(act_mesh, 
#               q_coll[0:n_coords], qdot_coll[0:n_coords], qddot_coll[0:n_coords],
#               q_coll[n_coords:2*n_coords], qdot_coll[n_coords:2*n_coords], qddot_coll[n_coords:2*n_coords],
#               qddot_coll[2*n_coords:3*n_coords], # accelerations are still controls
#               zk)
f_objk = Function('f_objk', [xk, uk], [f_objk_orig(xk, uk_orig)])
f_g0 = Function('f_g0', [xk, uk], [vertcat(f_g0_orig(xk, uk_orig), periodicity_constraint)])
f_gk = Function('f_gk', [xk, uk], [vertcat(f_gk_orig(xk, uk_orig), periodicity_constraint)])
f_gK = Function('f_gK', [xk], [f_gK_orig(xk)])
f_gk_ineq = Function('f_gk_ineq', [xk, uk], [f_gk_ineq_orig(xk, uk_orig)])
f_gap = Function('f_gap', [xk, uk], [zk if introduce_periodicity_vars else vertcat(zk, xk[n_coords*2:])])
f_g0_unique = Function('f_g0_unique', [xk, uk], [f_g0_orig_unique(xk, uk_orig)])

###############################
### Update path constraints ###
###############################
reformulator = Reformulator(n_coll, n_coords, n_act_mesh, dt, xk, uk, uk_orig, q_coll, qdot_coll, qddot_coll)
f_gk_reformulated = reformulator.get_f_gk_reformulated(f_gk)
f_g0_reformulated = Function('f_g0_reformulated', [xk, uk], [vertcat(f_g0_unique(xk, uk), f_gk_reformulated(xk, uk))])
# reformulator.show_gk_jacobian_structure_for_debugging(f_gk)
# reformulator.show_gk_jacobian_structure_for_debugging(f_gk_reformulated)

########################################
### update initial guess of controls ###
########################################
act_init = uu_init[:n_act_mesh, :]
q_coll_init = uu_init[n_act_mesh:n_act_mesh+n_coll*n_coords, :]
qdot_coll_init = uu_init[n_act_mesh+n_coll*n_coords:n_act_mesh+2*n_coll*n_coords, :]
qddot_coll_init = uu_init[n_act_mesh+2*n_coll*n_coords:n_act_mesh+3*n_coll*n_coords, :]

# actual controls
uu_init_reordered = act_init

# collocation points
for i in range(n_coll-1):
    uu_init_reordered = vertcat(uu_init_reordered, q_coll_init[i*n_coords:(i+1)*n_coords, :], qdot_coll_init[i*n_coords:(i+1)*n_coords, :], qddot_coll_init[i*n_coords:(i+1)*n_coords, :])

# last collocation point qddot and zk values
uu_init_reordered = vertcat(uu_init_reordered, 
                            qddot_coll_init[(n_coll-1)*n_coords:n_coll*n_coords, :], 
                            q_coll_init[(n_coll-1)*n_coords:n_coll*n_coords, :], 
                            qdot_coll_init[(n_coll-1)*n_coords:n_coll*n_coords, :])
if introduce_periodicity_vars:
    uu_init_reordered = vertcat(uu_init_reordered, xx_init[n_coords * 2:, :-1]) # periodicity vars initial guess

assert uu_init_reordered.shape[0] == nu + (2*n_coords - 1 if introduce_periodicity_vars else 0)

###########################
### setup opti instance ###
###########################
opti = Opti()
xx = []
uu = []
for k in range(N):
    xx.append(opti.variable(nx, 1))
    uu.append(opti.variable(nu + (2*n_coords - 1 if introduce_periodicity_vars else 0), 1))
xx.append(opti.variable(nx, 1))  # add state at mesh point N

obj = 0
for k in range(N):
    # gap-closing constraint
    opti.subject_to(xx[k+1] == f_gap(xx[k], uu[k]))
    
    # # equality constraint
    if k == 0:
        # opti.subject_to(f_g0(xx[k], uu[k]) == 0)
        opti.subject_to(f_g0_reformulated(xx[k], uu[k]) == 0)
    else:
        # opti.subject_to(f_gk(xx[k], uu[k]) == 0)
        opti.subject_to(f_gk_reformulated(xx[k], uu[k]) == 0)

    # inequality constraint
    opti.subject_to(lb <= (f_gk_ineq(xx[k], uu[k]) <= ub))
    
    # initial guess
    opti.set_initial(xx[k], xx_init[:, k])
    opti.set_initial(uu[k], uu_init_reordered[:, k])
    
    # final constraints
    if k == N-1:
        opti.subject_to(f_gK(xx[k+1]) == 0)
        opti.set_initial(xx[k+1], xx_init[:, k+1])
        
    # objective
    obj += f_objk(xx[k], uu[k])
    
opti.minimize(obj)

# Note: Fatrop does not support CasADi callbacks, so we will capture stdout during solving
convergence_stats = None

opts_casadi = {
    'expand': True, 
    'detect_simple_bounds': True,
    'structure_detection': 'auto'
}
opts_fatrop = {
    'tol': 1e-5,
    'mu_init': 0.1,
    'max_iter': 300,
    'problem_type': problem_type,
}
if problem_type == 'accelerated_ocp_type':
    opts_fatrop['linsol_nb_of_dynamics_constraints'] = reformulator.number_of_constraints_with_zk
    opts_fatrop['linsol_nb_of_zk_vars'] = zk.shape[0]

opti.solver('fatrop', opts_casadi, opts_fatrop)

if save_opti_f:
    print(f"creating opti to function object")
    opti_f = opti.to_function(
        f'test_gait_shortcut_python_reformulated_{problem_type}', 
        [], [hcat(xx), hcat(uu)], [], ['xx', 'uu'])
    
    print(f"saving opti to function object")
    opti_f.save(f'casadi_funcs/test_gait_shortcut_python_reformulated_{problem_type}.casadi')

if solve and not load_solution:
    print(f"solving")
    
    if config['load_opti_f']:
        opti_f = Function.load(f'casadi_funcs/test_gait_shortcut_python_reformulated_{problem_type}.casadi')
        
        if config['log_convergence']:
            solver_output_file = f'stored_solutions/solver_output_{problem_type}{config["file_name_appendix"]}.txt'
            with open(solver_output_file, 'w') as f:
                old_stdout = sys.stdout
                sys.stdout = f
                try:
                    xx_sol, uu_sol = opti_f()
                    stats = opti_f.stats()
                finally:
                    sys.stdout = old_stdout
        else:
            xx_sol, uu_sol = opti_f()
            stats = opti_f.stats()
        
    elif save_opti_f:
        if config['log_convergence']:
            solver_output_file = f'stored_solutions/solver_output_{problem_type}{config["file_name_appendix"]}.txt'
            with open(solver_output_file, 'w') as f:
                old_stdout = sys.stdout
                sys.stdout = f
                try:
                    xx_sol, uu_sol = opti_f()
                    stats = opti_f.stats()
                finally:
                    sys.stdout = old_stdout
        else:
            xx_sol, uu_sol = opti_f()
            stats = opti_f.stats()
    else:
        if config['log_convergence']:
            solver_output_file = f'stored_solutions/solver_output_{problem_type}{config["file_name_appendix"]}.txt'
            with open(solver_output_file, 'w') as f:
                old_stdout = sys.stdout
                sys.stdout = f
                try:
                    sol = opti.solve()
                    xx_sol = sol.value(hcat(xx))
                    uu_sol = sol.value(hcat(uu))
                    stats = sol.stats()
                finally:
                    sys.stdout = old_stdout
        else:
            sol = opti.solve()
            xx_sol = sol.value(hcat(xx))
            uu_sol = sol.value(hcat(uu))
            stats = sol.stats()
    
    if store_solution:
        with open(f'stored_solutions/solution_gait_shortcut_reformulated_{problem_type}{config["file_name_appendix"]}.pkl', 'wb') as f:
            pkl.dump({'xx_sol': xx_sol, 'uu_sol': uu_sol, 'stats': stats}, f)
    
    if config['log_convergence']:
        print(f"Solver output saved to: {solver_output_file}")
            
    nb_iterations = stats['fatrop']['iterations_count']
    print(f"Search dir computation time per iteration: {stats['fatrop']['compute_sd_time']/nb_iterations*1000:.4f} ms")
    print(f"other time per iteration: {stats['fatrop']['time_total']/nb_iterations*1000:.4f} ms")
            
if load_solution:
    print(f"loading solution")
    with open(f'stored_solutions/solution_gait_shortcut_reformulated_{problem_type}{config["file_name_appendix"]}.pkl', 'rb') as f:
        data = pkl.load(f)
        xx_sol = data['xx_sol']
        uu_sol = data['uu_sol']
        stats = data['stats']

if (solve or load_solution) and visualize_solution:   
    visualize_gait_cycle(xx_sol, n_coords, N, dt, problem_type)