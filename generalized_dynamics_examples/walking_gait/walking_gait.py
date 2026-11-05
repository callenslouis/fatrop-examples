from casadi import *
import yaml
import pickle as pkl

from visualizer import visualize_gait_cycle
from opti_generator import OptiGenerator
from walking_gait_solver import WalkingGaitSolver

# scale_constraints: same row scaling of the equality constraints for all problem types (see OptiGenerator.generate).
# Without scaling, the implicit formulation diverges: the stage-wise LU rank decisions near the end of the horizon
# (pivots ~1e-5, close to fatrop's absolute tolerance linsol_lu_fact_tol) corrupt the inertia detection.
def build_opti(opti_generator, problem_type, scale_constraints=True):
    if problem_type == 'implicit_ocp_type':
        # no auxiliary controls for the last collocation point: implicit dynamics
        opti, xx, uu, opts_casadi, opts_fatrop = opti_generator.generate_implicit(scale_constraints)
    else:
        opti, xx, uu, opts_casadi, opts_fatrop = opti_generator.generate(scale_constraints)

    # add additional options if required
    if problem_type == 'accelerated_ocp_type':
        opts_fatrop['problem_type'] = 'accelerated_ocp_type'
        opts_fatrop['linsol_nb_of_dynamics_constraints'] = opti_generator.get_nb_dynamics_constraints()
        opts_fatrop['linsol_nb_of_zk_vars'] = opti_generator.get_nb_zk_vars()

    opti.solver('fatrop', opts_casadi, opts_fatrop)
    return opti, xx, uu


# problem_cache: optional dict that keeps the Opti problems (and their solvers) across calls.
# Most of the setup time is spent by CasADi when it builds the derivative functions
# (jacobian, hessian of the expanded SX graph) during the first opti.solve(). Repeated
# opti.solve() calls on the same Opti object reuse the solver and start from the same initial guess.
def solve(config=None, problem_cache=None):
    if config is None:
        # Read config file
        config_file = "config.yaml"
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)

    if problem_cache is None:
        problem_cache = {}
    if 'opti_generator' not in problem_cache:
        problem_cache['opti_generator'] = OptiGenerator()
    opti_generator = problem_cache['opti_generator']
    solver = WalkingGaitSolver(config['problem_type'], config['file_name_appendix'])


    if config['load_solution']:
        xx_sol, uu_sol, stats = solver.load_solution()
            
    elif config['load_opti_f']:
        xx_sol, uu_sol, stats = solver.load_opti_f()
        
    else:
        # Get Opti Problem (built once per problem type)
        if config['problem_type'] not in problem_cache:
            problem_cache[config['problem_type']] = build_opti(opti_generator, config['problem_type'],
                                                               config.get('scale_constraints', True))
        opti, xx, uu = problem_cache[config['problem_type']]

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
        
if __name__ == "__main__":
    solve()