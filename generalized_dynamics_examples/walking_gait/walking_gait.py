from casadi import *
import yaml
import pickle as pkl

from visualizer import visualize_gait_cycle
from opti_generator import OptiGenerator
from walking_gait_solver import WalkingGaitSolver

# Read config file
config_file = "config.yaml"
with open(config_file, 'r') as f:
    config = yaml.safe_load(f)

opti_generator = OptiGenerator()
solver = WalkingGaitSolver(config['problem_type'], config['file_name_appendix'])


if config['load_solution']:
    xx_sol, uu_sol, stats = solver.load_solution()
        
elif config['load_opti_f']:
    xx_sol, uu_sol, stats = solver.load_opti_f()
    
else:
    # Get Opti Problem
    opti, xx, uu, opts_casadi, opts_fatrop = opti_generator.generate()

    # add additional options if required
    if config['problem_type'] == 'accelerated_ocp_type':
        opts_fatrop['problem_type'] = 'accelerated_ocp_type'
        opts_fatrop['linsol_nb_of_dynamics_constraints'] = opti_generator.get_nb_dynamics_constraints()
        opts_fatrop['linsol_nb_of_zk_vars'] = opti_generator.get_nb_zk_vars()

    opti.solver('fatrop', opts_casadi, opts_fatrop)
    
    if config['save_opti_f']:
        xx_sol, uu_sol, stats = solver.save_opti_f(opti, xx, uu)
    else:
        xx_sol, uu_sol, stats = solver.solve_plain_opti(opti, xx, uu)


if config['store_solution'] and not config['load_solution']:
    solver.store_solution(xx_sol, uu_sol, stats)

nb_iterations = stats['fatrop']['iterations_count']
print(f"Search dir computation time per iteration: {stats['fatrop']['compute_sd_time']/nb_iterations*1000:.4f} ms")
print(f"other time per iteration: {stats['fatrop']['time_total']/nb_iterations*1000:.4f} ms")
                

if config['visualize_solution']:   
    visualize_gait_cycle(xx_sol, opti_generator.n_coords, opti_generator.N, 
                         opti_generator.dt, config['problem_type'])