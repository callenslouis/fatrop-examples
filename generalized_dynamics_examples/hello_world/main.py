from casadi import *
import numpy as np
import time

from helper import get_dynamics_and_path_constraint, print_timings, visualize_solution

def solve(use_acceleration, nb_agents=2, visualize=False):
    N = 50
    dt = 0.05

    initial_px = np.array([np.cos(2*np.pi*0.2*i) for i in range(nb_agents)])
    initial_py = np.array([np.sin(2*np.pi*0.2*i) for i in range(nb_agents)])
    
    terminal_px = np.array([np.cos(2*np.pi*0.2*i + np.pi/2) for i in range(nb_agents)])
    terminal_py = np.array([np.sin(2*np.pi*0.2*i + np.pi/2) for i in range(nb_agents)])

    ### define casadi functions
    gap_closing_constraint, path_constraint = get_dynamics_and_path_constraint(nb_agents, dt)

    ### define opti instance
    opti = Opti()

    # define variables
    xx = []; uu = []; zz = []
    for k in range(N+1):
        xx.append(opti.variable(4*nb_agents)) # px, py, vx, vy for each agent
        if k < N:
            zz.append(opti.variable(4*nb_agents)) # auxiliary variables for dynamics
            uu.append(opti.variable(2*nb_agents)) # ax, ay for each agent
        
    obj = 0
    for k in range(N):
        # dynamics constraints (backward euler integrator)
        # opti.subject_to(f_gap_closing(xx[k], uu[k], xx[k+1]) == 0) # --> not supported by fatrop!
        opti.subject_to(xx[k+1] == zz[k]) # --> dummy dynamics (fit required form)
        
        # path constraints
        if k == 0:
            for i in range(nb_agents):
                opti.subject_to(xx[0][i] == initial_px[i])
                opti.subject_to(xx[0][i + nb_agents] == initial_py[i])
        opti.subject_to(path_constraint(xx[k]) == 0)
            
        # enforce dynamics as path constraints
        opti.subject_to(gap_closing_constraint(xx[k], uu[k], zz[k]) == 0) # --> enforce any dynamics
        
        # control bounds
        opti.subject_to(opti.bounded(-5, uu[k], 5))
        
        # objective: minimize control effort
        obj += sumsqr(uu[k])
        
        opti.set_initial(xx[k], np.array([initial_px, initial_py, np.zeros(nb_agents), np.zeros(nb_agents)]).reshape(-1, 1))
        opti.set_initial(zz[k], np.array([initial_px, initial_py, np.zeros(nb_agents), np.zeros(nb_agents)]).reshape(-1, 1))
        if k == N-1:
            opti.set_initial(xx[k+1], np.array([initial_px, initial_py, np.zeros(nb_agents), np.zeros(nb_agents)]).reshape(-1, 1))

    # terminal position constraint
    for i in range(nb_agents):
        opti.subject_to(xx[N][i] == terminal_px[i])
        opti.subject_to(xx[N][i + nb_agents] == terminal_py[i])
    
    # terminal velocity constraint
    opti.subject_to(xx[N][2*nb_agents:3*nb_agents] == 0)
    opti.subject_to(xx[N][3*nb_agents:] == 0)

    opti.minimize(obj)

    #######################################################
    ### Pass options to exploit reformulation structure ###
    opts_fatrop = {'problem_type': 'accelerated_ocp_type' if use_acceleration else 'ocp_type'} # ocp_type is default
    opti.solver('fatrop', 
                {'expand':True, 'structure_detection': 'auto'},
                opts_fatrop)
    #######################################################

    sol = opti.solve()
    
    # visualize trajectories
    if visualize:
        visualize_solution(nb_agents, sol.value(hcat(xx)))

    return sol  



if __name__ == "__main__":
    
    nb_agents = 10
    nb_runs = 3
    visualize_last = True
    
    sols_normal = []
    sols_accelerated = []
    for i in range(nb_runs):
        print(f"Run {i+1}/{nb_runs} - Solving without acceleration")
        start_time = time.time()
        sol_normal = solve(use_acceleration=False, nb_agents=nb_agents)
        sols_normal.append(sol_normal)

        print(f"Run {i+1}/{nb_runs} - Solving with acceleration")
        start_time = time.time()
        sol_accelerated = solve(use_acceleration=True, nb_agents=nb_agents, visualize=(visualize_last and i == nb_runs - 1))
        sols_accelerated.append(sol_accelerated)
    

    print_timings(sols_normal, sols_accelerated)
