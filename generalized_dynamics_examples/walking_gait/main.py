from walking_gait import solve
from post_process_test_gait import print_minimal_table, print_comparison
import yaml

if __name__ == '__main__':
    nb_runs = 4 # average over a few runs
    
    # load config
    with open("config.yaml") as f:
        config = yaml.safe_load(f)
        
    # set default settings
    config['store_solution'] = False # skip first run (also absorbs the one-time setup cost)

    # build each Opti problem and solver only once, reuse it for all runs
    problem_cache = {}
    problem_types = ['ocp_type', 'accelerated_ocp_type', 'implicit_ocp_type']
    failed = set()
    
    # solve normal ocp type
    for i in range(nb_runs):
        if i > 0:
            config['store_solution'] = True
            
        config['file_name_appendix'] = f"_{i}"
        
        for problem_type in problem_types:
            if problem_type in failed:
                continue
            config['problem_type'] = problem_type
            try:
                solve(config, problem_cache)
            except RuntimeError as e:
                # e.g. the solver did not converge: skip this problem type in the remaining runs
                print(f"{problem_type} failed, it is skipped in the remaining runs: {str(e).splitlines()[-1]}")
                failed.add(problem_type)
        
    # print post_processing table
    print_minimal_table()