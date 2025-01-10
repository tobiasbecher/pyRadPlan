#%% External package import

import cyipopt as ipopt

#%% Class definition


class Ipopt:
    def __init__(
        self,
        n_vars,
        n_constr,
        problem_instance,
        lb_var,
        ub_var,
        lb_constr,
        ub_constr,
        max_iter=500,
        max_cpu_time=3000,
    ):

        self.nlp = ipopt.problem(
            n=n_vars,
            m=n_constr,
            problem_obj=problem_instance,
            lb=lb_var,
            ub=ub_var,
            cl=lb_constr,
            cu=ub_constr,
        )

        self.options = self.set_solver_options(max_iter, max_cpu_time)

    def __str__(self):
        pass

    def set_solver_options(self, max_iter, max_cpu_time):

        max_cpu_time = max_cpu_time if isinstance(max_cpu_time, float) else float(max_cpu_time)

        try:
            options = {
                "tol": 1e-10,
                "dual_inf_tol": 1e-4,
                "constr_viol_tol": 1e-4,
                "compl_inf_tol": 1e-4,
                "acceptable_iter": 5,
                "acceptable_tol": 1e10,
                "acceptable_constr_viol_tol": 1e-2,
                "acceptable_dual_inf_tol": 1e10,
                "acceptable_compl_inf_tol": 1e10,
                "acceptable_obj_change_tol": 1e-4,
                "max_iter": max_iter,
                "max_cpu_time": max_cpu_time,
                "mu_strategy": "adaptive",
                "hessian_approximation": "limited-memory",
                "limited_memory_max_history": 6,
                "limited_memory_initialization": "scalar2",
                "linear_solver": "ma57",
            }

            for key, val in options.items():
                self.nlp.addOption(key, val)

        except TypeError:
            options = {
                "tol": 1e-8,
                "dual_inf_tol": 1.0,
                "constr_viol_tol": 1e-4,
                "compl_inf_tol": 1e-4,
                "acceptable_iter": 3,
                "acceptable_tol": 1e10,
                "acceptable_constr_viol_tol": 1e10,
                "acceptable_dual_inf_tol": 1e10,
                "acceptable_compl_inf_tol": 1e10,
                "acceptable_obj_change_tol": 1e-3,
                "max_iter": max_iter,
                "max_cpu_time": max_cpu_time,
                "mu_strategy": "adaptive",
                "hessian_approximation": "limited-memory",
                "limited_memory_max_history": 6,
                "limited_memory_initialization": "scalar2",
                "linear_solver": "ma27",
            }

            for key, val in options.items():
                self.nlp.addOption(key, val)

        return options
