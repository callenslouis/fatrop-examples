from walking_gait import solve
from post_process_test_gait import print_minimal_table
import yaml

if __name__ == '__main__':
    nb_runs = 3 # average over a few runs
    
    # load config
    with open("config.yaml") as f:
        config = yaml.safe_load(f)
        
    # set default settings
    config['store_solution'] = True
    
    # solve normal ocp type
    for i in range(nb_runs):
        config['file_name_appendix'] = f"_{i}"
        
        config['problem_type'] = 'ocp_type'
        solve(config)
        
        config['problem_type'] = 'accelerated_ocp_type'
        config['file_name_appendix'] = f"_{i}"
        
    # print post_processing table
    print_minimal_table()
    