from casadi import *

def save_opti_f(opti, xx, uu, config):
    print(f"saving opti to function object")
    opti_f = Function('opti_f', [], [], [hcat(xx), hcat(uu)], ['xx_sol', 'uu_sol'])
    
    print(f"saving opti to function object")
    opti_f.save(f'casadi_funcs/opti_to_function_{config["problem_type"]}.casadi')
    return opti_f
    