from walking_gait import solve
from post_process_test_gait import print_minimal_table, print_comparison
import yaml

if __name__ == '__main__':
    nb_runs = 4 # average over a few runs
    
    # load config
    with open("config.yaml") as f:
        config = yaml.safe_load(f)
        
    # set default settings
    config['store_solution'] = False # skip first run
    
    # solve normal ocp type
    for i in range(nb_runs):
        if i > 0:
            config['store_solution'] = True
            
        config['file_name_appendix'] = f"_{i}"
        
        config['problem_type'] = 'ocp_type'
        solve(config)
        
        config['problem_type'] = 'accelerated_ocp_type'
        solve(config)
        
    # print post_processing table
    print_minimal_table()