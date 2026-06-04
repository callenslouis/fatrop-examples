from casadi import *
import matplotlib.pyplot as plt

def get_dynamics_and_path_constraint(nb_agents, dt):
    pxk = SX.sym('pxk', nb_agents)
    pyk = SX.sym('pyk', nb_agents)
    vxk = SX.sym('vxk', nb_agents)
    vyk = SX.sym('vyk', nb_agents)
    axk = SX.sym('axk', nb_agents)
    ayk = SX.sym('ayk', nb_agents)
    pxk_next = SX.sym('pxk_next', nb_agents)
    pyk_next = SX.sym('pyk_next', nb_agents)
    vxk_next = SX.sym('vxk_next', nb_agents)
    vyk_next = SX.sym('vyk_next', nb_agents)

    xk = vertcat(pxk, pyk, vxk, vyk)
    uk = vertcat(axk, ayk)
    xk_next = vertcat(pxk_next, pyk_next, vxk_next, vyk_next)

    # simple backward euler integrator dynamics
    gap_closing_constraint = Function('f_gap_closing', [xk, uk, xk_next],
                            [vertcat(
                                pxk_next - (pxk + dt*vxk_next),
                                pyk_next - (pyk + dt*vyk_next),
                                vxk_next - (vxk + dt*axk),
                                vyk_next - (vyk + dt*ayk)
                            )])

    # enforce all agents to remain at a 1m distance from the first agent
    path_constraint = Function('f_distance_constraint', [xk],
                                    [vertcat(*[
                                        (pxk[i])**2 + (pyk[i])**2 - 1 for i in range(nb_agents)
                                    ])]) 
    
    return gap_closing_constraint, path_constraint

def visualize_solution(nb_agents, xx_sol):
    px_sol = xx_sol[:nb_agents, :]
    py_sol = xx_sol[nb_agents:2*nb_agents, :]

    plt.figure()
    for i in range(nb_agents):
        plt.plot(px_sol[i, :], py_sol[i, :], label=f'agent {i}')
    plt.legend()
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Trajectories of agents')
    plt.axis('equal')
    plt.grid()
    plt.show()
    
def print_timings(sols_normal, sols_accelerated):
    # get average search direction times and total times from the solutions
    avg_sd_time_normal = np.mean([sol.stats()['fatrop']['compute_sd_time'] / sol.stats()['fatrop']['iterations_count'] * 1000 for sol in sols_normal])
    avg_total_time_normal = np.mean([sol.stats()['fatrop']['time_total'] / sol.stats()['fatrop']['iterations_count'] * 1000 for sol in sols_normal])
    avg_sd_time_accelerated = np.mean([sol.stats()['fatrop']['compute_sd_time'] / sol.stats()['fatrop']['iterations_count'] * 1000 for sol in sols_accelerated])
    avg_total_time_accelerated = np.mean([sol.stats()['fatrop']['time_total'] / sol.stats()['fatrop']['iterations_count'] * 1000 for sol in sols_accelerated])
    
    # print table with results
    print()
    print(f"{'KPI':<30} {'Normal':<20} {'Accelerated':<20} {'Improvement':<20}")
    print(f"{'-'*90}")
    print(f"{'Search Direction Time (ms)':<30} {avg_sd_time_normal:<20.4f} {avg_sd_time_accelerated:<20.4f} {((avg_sd_time_accelerated - avg_sd_time_normal) / avg_sd_time_normal * 100):<20.2f}%")
    print(f"{'Total Time (ms)':<30} {avg_total_time_normal:<20.4f} {avg_total_time_accelerated:<20.4f} {((avg_total_time_accelerated - avg_total_time_normal) / avg_total_time_normal * 100):<20.2f}%")