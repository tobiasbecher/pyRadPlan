from ._base_scalarization_strategies import ScalarizationStrategyBase

class WeightedSum(ScalarizationStrategyBase):
    name = "Weighted sum scalarization strategy"
    short_name = "weighted_sum"

    def __init__(self,
            evaluate_objectives,
            evaluate_constraints,
            evaluate_x_gradients,
            evaluate_jacobian,
            etc,
            parameters):

        pass

    def variable_lower_bounds():
        pass
    def variable_upper_bounds():
        pass

    def get_linear_constraints(self):
        pass

    def get_nonlinear_constraints(self):
        pass

    def evaluate_objective(x):
        pass

    def evaluate_constraints(x):
        pass

