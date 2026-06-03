import pickle as pkl
from casadi import *

class WalkingGaitSolver:
    def __init__(self, problem_type, file_name_appendix, log_convergence=False):
        self.problem_type = problem_type
        self.file_name_appendix = file_name_appendix
        self.log_convergence = log_convergence

    def load_solution(self):
        print(f"loading solution")
        with open(f'stored_solutions/solution_walking_gait_{self.problem_type}{self.file_name_appendix}.pkl', 'rb') as f:
            data = pkl.load(f)
            xx_sol = data['xx_sol']
            uu_sol = data['uu_sol']
            stats = data['stats']
            
        return xx_sol, uu_sol, stats
    
    def load_opti_f(self):
        opti_f = Function.load(f'casadi_funcs/opti_to_function_{self.problem_type}.casadi')
        
        return self.solve_opti_f_instance(opti_f)
    
    def save_opti_f(self, opti, xx, uu):
        print(f"saving opti to function object")
        opti_f = opti.to_function('opti_f', [], [], [hcat(xx), hcat(uu)], ['xx_sol', 'uu_sol'])
        
        print(f"saving opti to function object")
        opti_f.save(f'casadi_funcs/opti_to_function_{self.problem_type}.casadi')
        
        return self.solve_opti_f_instance(opti_f)
    
    def solve_plain_opti(self, opti, xx, uu):
        if self.log_convergence:
            solver_output_file = f'stored_solutions/solver_output_{self.problem_type}{self.file_name_appendix}.txt'
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
            
        return xx_sol, uu_sol, stats
        
    def solve_opti_f_instance(self, opti_f):
        if self.log_convergence:
            solver_output_file = f'stored_solutions/solver_output_{self.problem_type}{self.file_name_appendix}.txt'
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
            
        return xx_sol, uu_sol, stats
    
    def store_solution(self, xx_sol, uu_sol, stats):
        with open(f'stored_solutions/solution_walking_gait_{self.problem_type}{self.file_name_appendix}.pkl', 'wb') as f:
            pkl.dump({'xx_sol': xx_sol, 'uu_sol': uu_sol, 'stats': stats}, f)